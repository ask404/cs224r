#!/usr/bin/env python3
"""Proposal stretch metric: calibrate the head's predicted joint distribution against
the GROUND-TRUTH empirical distribution from K=64 rollouts per prompt (paper Table 1).

For each prompt (a group of K rollouts):
  - EMPIRICAL joint  : histogram of the K rollouts' (reward_bin, total_length_bin),
                       using ziprc.grid binning (length = full response length, since at
                       the prompt prefix the remaining length IS the total length).
  - PREDICTED joint  : the head's softmax over its reserved slice at the prompt's last
                       token (one forward pass per prompt).
  - Total Variation  : 0.5 * sum |empirical - predicted|.

Also reports END-of-generation reward prediction (paper's F1/accuracy at threshold 0.5)
and a LENGTH-calibration check (predicted E[total length] vs actual mean), validating the
COST half of the joint reward-cost prediction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ziprc.alignment import train_time_align  # noqa: E402
from ziprc.config import (DISTRIBUTION_TOKEN_ID, LENGTH_BINS_COUNTDOWN,
                          REWARD_VALUES_BINARY)  # noqa: E402
from ziprc.grid import bin_index, length_bin_of  # noqa: E402


def _to_int_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(t) for t in x]
    return [int(t) for t in list(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in-parquet", required=True, help="K-per-prompt labeled rollouts (needs 'correct').")
    ap.add_argument("--distribution-token-id", type=int, default=DISTRIBUTION_TOKEN_ID)
    ap.add_argument("--reward-values", type=float, nargs="+", default=list(REWARD_VALUES_BINARY))
    ap.add_argument("--length-bins", type=int, nargs="+", default=list(LENGTH_BINS_COUNTDOWN))
    ap.add_argument("--min-k", type=int, default=32, help="Only score prompts with >= this many rollouts.")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = len(args.reward_values)
    L = len(args.length_bins) - 1
    num_bins = R * L
    start = args.distribution_token_id
    rv = np.asarray(args.reward_values, float)
    length_mid = np.array([0.5 * (args.length_bins[i] + args.length_bins[i + 1]) for i in range(L)])

    df = pd.read_parquet(args.in_parquet)
    groups = [(pi, g) for pi, g in df.groupby("prompt_idx") if len(g) >= args.min_k]
    print(f"[calib_tv] {len(groups)} prompts with >= {args.min_k} rollouts | R={R} L={L} bins={num_bins}")

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2").to(device).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    weight_bins = lm_head.weight[start:start + num_bins]
    bias_bins = lm_head.bias[start:start + num_bins] if getattr(lm_head, "bias", None) is not None else None

    tvs, pred_len_err = [], []
    y_true_end, y_pred_end = [], []
    with torch.inference_mode():
        for pi, g in groups:
            # prompt = tokens before the first labelled (response) position
            row0 = g.iloc[0]
            ids0 = _to_int_list(row0["input_ids"])
            lp0 = _to_int_list(row0["label_positions"])
            prompt_len = lp0[0] if lp0 else len(ids0)
            prompt_ids = ids0[:prompt_len]
            if len(prompt_ids) < 1:
                continue

            # PREDICTED joint at the prompt's last token
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            h = model(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states[-1][0, -1]
            logits = F.linear(h.float(), weight_bins.float(), None if bias_bins is None else bias_bins.float())
            pred = F.softmax(logits, dim=-1).view(R, L).cpu().numpy()  # [R, L]

            # EMPIRICAL joint over the K rollouts (reward, total response length)
            emp = np.zeros((R, L))
            lens = []
            for _, row in g.iterrows():
                resp_len = int(row["length"])
                lens.append(resp_len)
                reward = float(row["correct"])
                idx = bin_index(resp_len, reward, args.reward_values, args.length_bins)
                emp[idx // L, idx % L] += 1
                y_true_end.append(reward)
            emp /= emp.sum()

            tvs.append(0.5 * np.abs(emp - pred).sum())
            # length calibration: predicted E[len] (length marginal) vs actual mean
            pred_len = float((pred.sum(axis=0) @ length_mid))
            pred_len_err.append(abs(pred_len - float(np.mean(lens))))

            # end-of-gen reward prediction: predicted P(correct) at prompt = pred reward marginal
            p_correct = float(pred.sum(axis=1) @ rv)
            y_pred_end.extend([p_correct] * len(g))

    tvs = np.array(tvs)
    yt, yp = np.array(y_true_end), np.array(y_pred_end)
    pred_pos = (yp >= 0.5)
    tp = float(((pred_pos == 1) & (yt == 1)).sum()); fp = float(((pred_pos == 1) & (yt == 0)).sum())
    fn = float(((pred_pos == 0) & (yt == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = float((pred_pos == (yt >= 0.5)).mean()) if len(yt) else float("nan")

    res = {
        "n_prompts": len(tvs),
        "mean_TV": float(tvs.mean()), "median_TV": float(np.median(tvs)),
        "end_reward_F1": f1, "end_reward_acc": acc,
        "length_MAE": float(np.mean(pred_len_err)),
    }
    print("\n=== K-rollout calibration (paper Table 1 style) ===")
    for k, v in res.items():
        print(f"  {k:<16} {v:.4f}" if isinstance(v, float) else f"  {k:<16} {v}")
    print("  (paper ZIP-RC-Lite ref: begin-TV ~0.63, end F1 ~0.82, acc ~0.71)")

    if args.out_json:
        import json
        json.dump(res, open(args.out_json, "w"), indent=2)
        print(f"[calib_tv] wrote {args.out_json}")


if __name__ == "__main__":
    main()
