#!/usr/bin/env python3
"""Blend test: does **adaptive-K** (across-prompt allocation) compose with **prune**
(within-sample compute) so the savings *compound*?

The two levers act on different quantities:
  * adaptive-K changes ORACLE at ~matched sample budget (spend whole samples where they help);
  * prune changes COST (tokens) at ~matched oracle (kill predicted-loser streams mid-generation).
"Compounding" means the blend captures BOTH, ~independently: prune saves the same token-fraction
under either allocation, AND adaptive's oracle lift survives pruning (it doesn't kill the very
frontier samples adaptive paid for — failure mode #4).

Method (faithful + cheap, ONE generation per prompt):
  1. Generate a pool of `pool_k` samples with policy="none" (no real pruning), recording each
     sample's value trajectory `vhist`, length, correctness, and the 25%-through value.
  2. Allocate ONLINE from THIS pool's probe (first probe_k samples) via the SAME `allocate()`
     the offline allocator uses — so adaptive is self-consistent (no stale allocation).
  3. Replay the EXACT decode-time prune rule OFFLINE over a threshold sweep to fill the full
     adaptive x prune 2x2 per threshold. Faithful because token-streams are independent
     (pruning sample i never changes j), so truncating i at its prune-point is exact.
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
from ziprc.allocate_budget import allocate  # noqa: E402
from ziprc.config import (DISTRIBUTION_TOKEN_ID, LENGTH_BINS_COUNTDOWN,  # noqa: E402
                          REWARD_VALUES_BINARY)


def sim(sel, prune, warmup, interval, keep_min, w, thr):
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
                    if sv[i] < thr:                        # only abandon predicted-losers
                        pruned[i] = True
    solved = any((not pruned[i]) and sel[i]["correct"] for i in range(K))
    return int(cost), bool(solved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Trained head-only model dir.")
    ap.add_argument("--prompts", required=True, help="Parquet with one row/prompt (prompt,target,nums[,prompt_idx]).")
    ap.add_argument("--probe-k", type=int, default=2)
    ap.add_argument("--pool-k", type=int, default=8, help="Samples generated per prompt (>= kmax).")
    ap.add_argument("--budget", type=float, default=6.0, help="Target mean total samples/prompt for adaptive.")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--scheme", choices=["frontier", "promise"], default="frontier")
    ap.add_argument("--num-prompts", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--prune-interval", type=int, default=32)
    ap.add_argument("--keep-min", type=int, default=2)
    ap.add_argument("--smooth-w", type=int, default=4)
    ap.add_argument("--prune-thresholds", type=float, nargs="+", default=[0.5, 0.4, 0.3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="If set, repeat the whole online run at each seed and report mean±std "
                         "+ per-seed Pareto-dominance count (firms up small-n pools).")
    ap.add_argument("--out", default=None, help="Threshold-sweep summary parquet.")
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

    df = pd.read_parquet(args.prompts)
    if "prompt_idx" in df.columns:
        df = df.drop_duplicates("prompt_idx")
    df = df.reset_index(drop=True)
    if "target" not in df.columns or "nums" not in df.columns:   # main-pool parquet stores ground_truth
        gt = df["ground_truth"]
        df = df.assign(
            target=[int(g["target"]) for g in gt],
            nums=[[int(x) for x in (g["numbers"] if "numbers" in g else g["nums"])] for g in gt],
        )
    if args.num_prompts and args.num_prompts < len(df):
        df = df.iloc[: args.num_prompts].reset_index(drop=True)

    def run_seed(seed):
        """One full online run at `seed`: generate pools, allocate from the fresh probe, and
        return per-(tau, arm, prune) mean oracle & cost plus allocation info."""
        pools, promise, solved = [], [], []
        for idx, r in enumerate(df.itertuples(index=False)):
            gt = {"target": int(r.target), "numbers": [int(x) for x in r.nums]}
            torch.manual_seed(seed * 100003 + int(getattr(r, "prompt_idx", idx)))
            o = dec.decode_prompt(r.prompt, args.pool_k, args.max_new_tokens, args.temperature,
                                  args.top_p, args.top_k, policy="none",
                                  warmup=args.warmup, prune_interval=args.prune_interval,
                                  keep_min=args.keep_min, smooth_w=args.smooth_w)
            pool = []
            for s in o["samples"]:
                vh = s["vhist"]
                q25 = vh[len(vh) // 4] if vh else 0.0        # value 25% through the trajectory
                pool.append({"vhist": vh, "n_tokens": s["n_tokens"],
                             "correct": compute_score(s["text"], gt) == 1.0, "q25": q25})
            pools.append(pool)
            probe = pool[: args.probe_k]
            promise.append(float(np.mean([p["q25"] for p in probe])) if probe else 0.0)
            solved.append(bool(any(p["correct"] for p in probe)))
        n_extra = allocate(promise, solved, args.budget, args.probe_k, args.kmax, args.scheme)
        K_i = args.probe_k + n_extra.astype(int)
        res = {"K_f": int(round(K_i.mean())), "meanK": float(K_i.mean()), "solved": int(np.sum(solved))}
        K_f = res["K_f"]
        for thr in args.prune_thresholds:
            cells = {(a, p): {"cost": [], "orc": []} for a in ("fixed", "adaptive") for p in (False, True)}
            for j, pool in enumerate(pools):
                for arm, K in (("fixed", K_f), ("adaptive", int(K_i[j]))):
                    selp = pool[:K]
                    for prune in (False, True):
                        c, s = sim(selp, prune, args.warmup, args.prune_interval, args.keep_min, args.smooth_w, thr)
                        cells[(arm, prune)]["cost"].append(c)
                        cells[(arm, prune)]["orc"].append(float(s))
            for a in ("fixed", "adaptive"):
                for p in (False, True):
                    res[(thr, a, p, "orc")] = float(np.mean(cells[(a, p)]["orc"]))
                    res[(thr, a, p, "cost")] = float(np.mean(cells[(a, p)]["cost"]))
        return res

    seeds = args.seeds if args.seeds else [args.seed]
    runs = []
    for si, sd in enumerate(seeds):
        runs.append(run_seed(sd))
        print(f"[blend] seed {sd} done ({si + 1}/{len(seeds)}) | K_fixed={runs[-1]['K_f']} "
              f"meanK={runs[-1]['meanK']:.2f} probe-solved={runs[-1]['solved']}/{len(df)}", flush=True)

    def agg(thr, a, p, field):
        v = [r[(thr, a, p, field)] for r in runs]
        return float(np.mean(v)), float(np.std(v))

    print(f"\n=== BLEND multi-seed: adaptive-K x prune ({len(seeds)} seeds, n={len(df)} prompts) ===")
    summary = []
    for thr in args.prune_thresholds:
        bf_o, bf_os = agg(thr, "fixed", False, "orc")
        bf_c, _ = agg(thr, "fixed", False, "cost")
        print(f"\n--- tau={thr:g} ---  (cost shown vs baseline fixed+full = {bf_c:.0f})")
        print(f"  {'config':<16} {'oracle mean±std':>16} {'cost':>8} {'vs base':>9}")
        for a in ("fixed", "adaptive"):
            for p in (False, True):
                o, os_ = agg(thr, a, p, "orc")
                c, _ = agg(thr, a, p, "cost")
                print(f"  {a + '+' + ('prune' if p else 'full'):<16} {o:7.3f} ± {os_:.3f}   "
                      f"{c:8.0f} {(1 - c / bf_c) * 100:+8.1f}%")
        # paired per-seed dominance: same-seed adaptive+prune oracle >= fixed+full AND cheaper
        dom = [(r[(thr, "adaptive", True, "orc")] >= r[(thr, "fixed", False, "orc")] - 1e-9)
               and (r[(thr, "adaptive", True, "cost")] < r[(thr, "fixed", False, "cost")]) for r in runs]
        ap_o, ap_os = agg(thr, "adaptive", True, "orc")
        ap_c, _ = agg(thr, "adaptive", True, "cost")
        print(f"  >>> blend(adaptive+prune): oracle {ap_o:.3f}±{ap_os:.3f} at {(1 - ap_c / bf_c) * 100:+.1f}% compute "
              f"| Pareto-dominates in {sum(dom)}/{len(seeds)} seeds")
        summary.append({"tau": thr, "n_seeds": len(seeds), "n_prompts": len(df),
                        "fixed_full_orc": bf_o, "fixed_full_orc_std": bf_os, "fixed_full_cost": bf_c,
                        "adapt_prune_orc": ap_o, "adapt_prune_orc_std": ap_os, "adapt_prune_cost": ap_c,
                        "pareto_seeds": int(sum(dom))})

    if args.out:
        pd.DataFrame(summary).to_parquet(args.out, index=False)
        print(f"\n[blend] wrote multi-seed summary -> {args.out}")


if __name__ == "__main__":
    main()
