#!/usr/bin/env python3
"""M5/M6: score rollouts with a trained ZIP-RC head (offline, teacher-forced).

Single forward pass per sequence (NO token-by-token decoding, NO logit masking —
the backbone is frozen so hidden states equal the policy's). Reads the reserved
slice, softmaxes the joint, marginalizes over length to P(reward_state), and
writes per-rollout value columns used by ziprc/value_select.py:

  value_mean : mean expected reward over all response positions
  value_end  : mean expected reward over the last --end-k response positions
               (the terminal-reward signal used for Best-of-N selection)

Adapted from github.com/rohinmanvi/ZIP-RC (src/score_with_ziprc_joint_head.py),
simplified to single-GPU since the 0.5B model + short Countdown rollouts are cheap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ziprc.alignment import train_time_align  # noqa: E402
from ziprc.config import (DISTRIBUTION_TOKEN_ID, LENGTH_BINS_COUNTDOWN,
                          REWARD_VALUES_STRUCTURED)  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Trained head-only model dir.")
    p.add_argument("--in-parquet", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--distribution-token-id", type=int, default=DISTRIBUTION_TOKEN_ID)
    p.add_argument("--reward-values", type=float, nargs="+", default=list(REWARD_VALUES_STRUCTURED))
    p.add_argument("--length-bins", type=int, nargs="+", default=list(LENGTH_BINS_COUNTDOWN))
    p.add_argument("--end-k", type=int, default=16, help="value_end averages the last K positions.")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    R = len(args.reward_values)
    L = len(args.length_bins) - 1
    num_bins = R * L
    start = args.distribution_token_id
    reward_values = torch.tensor(args.reward_values, dtype=torch.float32, device=device)

    df = pd.read_parquet(args.in_parquet)
    print(f"[score] {len(df)} rollouts | R={R} L={L} bins={num_bins} start={start}", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation="flash_attention_2",
            trust_remote_code=True).to(device).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    assert start + num_bins <= lm_head.weight.size(0), "reserved slice exceeds vocab"
    weight_bins = lm_head.weight[start: start + num_bins]
    bias_bins = (lm_head.bias[start: start + num_bins]
                 if getattr(lm_head, "bias", None) is not None else None)

    # value_end  -> EASY case: near end of generation (answer already written) -> selection
    # value_first/_q25 -> HARD case: early prefix -> the real test for mid-gen pruning
    value_mean, value_end, value_first, value_q25 = [], [], [], []
    with torch.inference_mode():
        for _, row in df.iterrows():
            input_ids = list(row["input_ids"])
            label_positions = list(row["label_positions"])
            ids, pos, _ = train_time_align(input_ids, label_positions, max_length=args.max_length)
            if not pos:
                for lst in (value_mean, value_end, value_first, value_q25):
                    lst.append(float("nan"))
                continue
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            h = model(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
            hp = h[torch.tensor(pos, device=device)]                     # [P, E]
            logits = F.linear(hp.float(), weight_bins.float(),
                              None if bias_bins is None else bias_bins.float())  # [P, bins]
            probs = F.softmax(logits, dim=-1).view(-1, R, L)
            p_reward = probs.sum(dim=-1)                                  # [P, R]
            exp_val = p_reward @ reward_values                           # [P], ordered by position
            P = exp_val.shape[0]
            value_mean.append(float(exp_val.mean().item()))
            value_end.append(float(exp_val[-args.end_k:].mean().item()))
            value_first.append(float(exp_val[0].item()))                 # first response prefix
            value_q25.append(float(exp_val[min(P - 1, P // 4)].item()))  # ~25% through

    df["value_mean"] = value_mean
    df["value_end"] = value_end
    df["value_first"] = value_first
    df["value_q25"] = value_q25
    df.to_parquet(args.out_parquet, engine="pyarrow", index=False)
    print(f"[score] wrote {args.out_parquet}", flush=True)

    # Quick calibration read if labels present
    for col in ("correct", "reward3"):
        if col in df.columns:
            m = df["value_end"].notna()
            y = df.loc[m, col].astype(float).values
            v = df.loc[m, "value_end"].values
            if len(y) > 1 and len(set(y.tolist())) > 1:
                corr = float(np.corrcoef(y, v)[0, 1])
                print(f"[score] corr(value_end, {col}) = {corr:.3f}", flush=True)


if __name__ == "__main__":
    main()
