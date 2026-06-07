#!/usr/bin/env python3
"""Blend test: does **adaptive-K** (across-prompt allocation) compose with **prune**
(within-sample compute) so the savings *compound*?

The two levers act on different quantities:
  * adaptive-K changes ORACLE at ~matched sample budget (spend whole samples where they help);
  * prune changes COST (tokens) at ~matched oracle (kill predicted-loser streams mid-generation).
"Compounding" means the blend captures BOTH, ~independently: prune saves the same token-fraction
under either allocation, AND adaptive's oracle lift survives pruning (it doesn't kill the very
frontier samples adaptive paid for — failure mode #4).

Method (faithful + cheap): for each prompt we generate a pool of K_gen samples ONCE with
`policy="none"` (no real pruning), recording each sample's value trajectory `vhist` and length.
Because samples are independent token-streams, pruning sample i never changes sample j — so the
full adaptive×prune 2×2 can be replayed OFFLINE from that single generation by (a) subsetting the
pool (first K_f = fixed, first K_i = adaptive) and (b) re-running the EXACT decode-time prune rule
on the recorded `vhist`. Cost = active forward passes (identical accounting to `adaptive_decode`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from evaluation.countdown import compute_score  # noqa: E402
from ziprc.adaptive_decode import AdaptiveDecoder, _smooth  # noqa: E402
from ziprc.config import (DISTRIBUTION_TOKEN_ID, LENGTH_BINS_COUNTDOWN,  # noqa: E402
                          REWARD_VALUES_BINARY)


def sim(sel, prune, warmup, interval, keep_min, w):
    """Replay decode_prompt's loop on pre-recorded trajectories for a selected sample subset.
    sel: list of dicts {vhist:list, n_tokens:int(=len vhist), correct:bool}.
    Returns (cost = active forward passes, solved = any surviving-correct)."""
    K = len(sel)
    if K == 0:
        return 0, False
    T = [s["n_tokens"] for s in sel]
    vh = [s["vhist"] for s in sel]
    if not prune:
        return int(sum(T)), bool(any(s["correct"] for s in sel))
    pruned = [False] * K
    cost = 0
    for step in range(max(T)):
        act = [i for i in range(K) if step < T[i] and not pruned[i]]
        if not act:
            break
        cost += len(act)                                   # one forward pass per active sample
        if step >= warmup and step % interval == 0:
            # decode-time prune candidates = active AND still generating (not on their done token)
            elig = [i for i in act if step < T[i] - 1]
            if len(elig) > keep_min:
                sv = {i: _smooth(vh[i][: step + 1], w) for i in elig}
                for i in sorted(elig, key=lambda j: sv[j])[: len(elig) - keep_min]:
                    if sv[i] < 0.5:                         # only abandon predicted-losers
                        pruned[i] = True
    solved = any((not pruned[i]) and sel[i]["correct"] for i in range(K))
    return int(cost), bool(solved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Trained head-only model dir.")
    ap.add_argument("--alloc", required=True, help="allocate_budget output (prompt,target,nums,ground_truth,n_extra).")
    ap.add_argument("--probe-k", type=int, default=2)
    ap.add_argument("--num-prompts", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--prune-interval", type=int, default=32)
    ap.add_argument("--keep-min", type=int, default=2)
    ap.add_argument("--smooth-w", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Per-prompt parquet for follow-up analysis.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to(dev).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    dec = AdaptiveDecoder(model, tok, DISTRIBUTION_TOKEN_ID, list(REWARD_VALUES_BINARY),
                          list(LENGTH_BINS_COUNTDOWN), dev)

    alloc = pd.read_parquet(args.alloc)
    if args.num_prompts and args.num_prompts < len(alloc):
        alloc = alloc.iloc[: args.num_prompts].reset_index(drop=True)
    alloc["K_i"] = args.probe_k + alloc["n_extra"].astype(int)
    K_f = int(round(alloc["K_i"].mean()))                  # fixed baseline = mean adaptive budget
    n_solved_skip = int((alloc["n_extra"].astype(int) == 0).sum())
    print(f"[blend] {len(alloc)} prompts | probe_k={args.probe_k} K_fixed={K_f} "
          f"mean K_i={alloc['K_i'].mean():.2f} | {n_solved_skip} probe-skipped (K_i={args.probe_k})", flush=True)

    cells = {(a, p): {"cost": [], "orc": []} for a in ("fixed", "adaptive") for p in (False, True)}
    rows = []
    for r in alloc.itertuples(index=False):
        K_i = int(r.K_i)
        K_gen = max(K_f, K_i)
        gt = {"target": int(r.target), "numbers": [int(x) for x in r.nums]}
        torch.manual_seed(args.seed + int(r.prompt_idx))
        out = dec.decode_prompt(r.prompt, K_gen, args.max_new_tokens, args.temperature,
                                args.top_p, args.top_k, policy="none",
                                warmup=args.warmup, prune_interval=args.prune_interval,
                                keep_min=args.keep_min, smooth_w=args.smooth_w)
        pool = [{"vhist": s["vhist"], "n_tokens": s["n_tokens"],
                 "correct": compute_score(s["text"], gt) == 1.0} for s in out["samples"]]
        rec = {"prompt_idx": int(r.prompt_idx), "tier": len(r.nums), "K_i": K_i, "K_f": K_f}
        for arm, K in (("fixed", K_f), ("adaptive", K_i)):
            sel = pool[:K]
            for prune in (False, True):
                c, s = sim(sel, prune, args.warmup, args.prune_interval, args.keep_min, args.smooth_w)
                cells[(arm, prune)]["cost"].append(c)
                cells[(arm, prune)]["orc"].append(float(s))
                rec[f"{arm}_{'prune' if prune else 'full'}_cost"] = c
                rec[f"{arm}_{'prune' if prune else 'full'}_solved"] = int(s)
        rows.append(rec)

    def mc(a, p):
        return float(np.mean(cells[(a, p)]["cost"]))

    def mo(a, p):
        return float(np.mean(cells[(a, p)]["orc"]))

    base = mc("fixed", False)
    print("\n=== BLEND 2x2: adaptive-K (allocation) x prune (within-sample compute) ===")
    print(f"  {'config':<20} {'oracle':>7} {'cost':>8}  {'vs fixed+full':>14}")
    for a in ("fixed", "adaptive"):
        for p in (False, True):
            tag = f"{a} + {'prune' if p else 'full '}"
            print(f"  {tag:<20} {mo(a, p):7.3f} {mc(a, p):8.1f}  {(1 - mc(a, p) / base) * 100:+13.1f}%")

    p_f = 1 - mc("fixed", True) / mc("fixed", False)
    p_a = 1 - mc("adaptive", True) / mc("adaptive", False)
    lift_full = mo("adaptive", False) - mo("fixed", False)
    lift_prune = mo("adaptive", True) - mo("fixed", True)
    hit_f = mo("fixed", False) - mo("fixed", True)
    hit_a = mo("adaptive", False) - mo("adaptive", True)
    blend_save = 1 - mc("adaptive", True) / base
    predicted = 1 - (mc("adaptive", False) / base) * (mc("fixed", True) / mc("fixed", False))
    print("\n=== COMPOUNDING ANALYSIS ===")
    print(f"  prune token-saving:   fixed {p_f * 100:+.1f}%   adaptive {p_a * 100:+.1f}%   "
          f"(close => prune is allocation-agnostic)")
    print(f"  prune oracle-hit:     fixed {hit_f:+.3f}      adaptive {hit_a:+.3f}      (smaller is safer)")
    print(f"  adaptive oracle-lift: full {lift_full:+.3f}   pruned {lift_prune:+.3f}   "
          f"(lift survives prune? => levers don't fight)")
    print(f"  blend cost-saving vs baseline: {blend_save * 100:+.1f}%   "
          f"(product-of-levers predicts {predicted * 100:+.1f}%)")
    dom = (mc("adaptive", True) < base) and (mo("adaptive", True) >= mo("fixed", False) - 1e-9)
    print(f"  >>> blend Pareto-dominates fixed+full (cheaper AND >= oracle): {dom}")

    if args.out:
        pd.DataFrame(rows).to_parquet(args.out, index=False)
        print(f"[blend] wrote per-prompt -> {args.out}")


if __name__ == "__main__":
    main()
