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

    df = pd.read_parquet(args.prompts)
    if "prompt_idx" in df.columns:
        df = df.drop_duplicates("prompt_idx")
    df = df.reset_index(drop=True)
    if args.num_prompts and args.num_prompts < len(df):
        df = df.iloc[: args.num_prompts].reset_index(drop=True)

    # --- ONE generation per prompt: pool_k samples, record vhist/len/correct/value_q25 ---
    pools, promise, solved = [], [], []
    for idx, r in enumerate(df.itertuples(index=False)):
        gt = {"target": int(r.target), "numbers": [int(x) for x in r.nums]}
        torch.manual_seed(args.seed + int(getattr(r, "prompt_idx", idx)))
        out = dec.decode_prompt(r.prompt, args.pool_k, args.max_new_tokens, args.temperature,
                                args.top_p, args.top_k, policy="none",
                                warmup=args.warmup, prune_interval=args.prune_interval,
                                keep_min=args.keep_min, smooth_w=args.smooth_w)
        pool = []
        for s in out["samples"]:
            vh = s["vhist"]
            q25 = vh[len(vh) // 4] if vh else 0.0            # value 25% through the trajectory
            pool.append({"vhist": vh, "n_tokens": s["n_tokens"],
                         "correct": compute_score(s["text"], gt) == 1.0, "q25": q25})
        pools.append(pool)
        probe = pool[: args.probe_k]
        promise.append(float(np.mean([p["q25"] for p in probe])) if probe else 0.0)
        solved.append(bool(any(p["correct"] for p in probe)))
        if (idx + 1) % 20 == 0:
            print(f"[blend] generated {idx + 1}/{len(df)} prompts", flush=True)

    # --- ONLINE allocation from THIS pool's fresh probe (identical logic to allocate_budget) ---
    n_extra = allocate(promise, solved, args.budget, args.probe_k, args.kmax, args.scheme)
    K_i = args.probe_k + n_extra.astype(int)
    K_f = int(round(K_i.mean()))
    print(f"[blend] {len(df)} prompts | probe_k={args.probe_k} pool_k={args.pool_k} "
          f"K_fixed={K_f} mean K_i={K_i.mean():.2f} | {int(np.sum(solved))} probe-solved (fresh)", flush=True)

    base = None
    summary = []
    for thr in args.prune_thresholds:
        cells = {(a, p): {"cost": [], "orc": []} for a in ("fixed", "adaptive") for p in (False, True)}
        for j, pool in enumerate(pools):
            for arm, K in (("fixed", K_f), ("adaptive", int(K_i[j]))):
                selp = pool[:K]
                for prune in (False, True):
                    c, s = sim(selp, prune, args.warmup, args.prune_interval, args.keep_min, args.smooth_w, thr)
                    cells[(arm, prune)]["cost"].append(c)
                    cells[(arm, prune)]["orc"].append(float(s))

        def mc(a, p):
            return float(np.mean(cells[(a, p)]["cost"]))

        def mo(a, p):
            return float(np.mean(cells[(a, p)]["orc"]))

        if base is None:
            base = mc("fixed", False)
            print(f"\n[ baseline fixed+full: oracle {mo('fixed', False):.3f}  cost {base:.1f} ]")
        print(f"\n=== prune threshold tau={thr:g} ===")
        print(f"  {'config':<20} {'oracle':>7} {'cost':>8}  {'vs base':>9}")
        for a in ("fixed", "adaptive"):
            for p in (False, True):
                tag = f"{a} + {'prune' if p else 'full '}"
                print(f"  {tag:<20} {mo(a, p):7.3f} {mc(a, p):8.1f}  {(1 - mc(a, p) / base) * 100:+8.1f}%")
        p_f = 1 - mc("fixed", True) / mc("fixed", False)
        p_a = 1 - mc("adaptive", True) / mc("adaptive", False)
        lift_full = mo("adaptive", False) - mo("fixed", False)
        lift_prune = mo("adaptive", True) - mo("fixed", True)
        dom = (mc("adaptive", True) < base) and (mo("adaptive", True) >= mo("fixed", False) - 1e-9)
        print(f"  prune token-save: fixed {p_f * 100:+.1f}% / adapt {p_a * 100:+.1f}%   "
              f"adaptive lift: full {lift_full:+.3f} / pruned {lift_prune:+.3f}   "
              f"blend Pareto-dominates: {dom}")
        summary.append({"tau": thr, "fixed_full_orc": mo("fixed", False), "fixed_prune_orc": mo("fixed", True),
                        "adapt_full_orc": mo("adaptive", False), "adapt_prune_orc": mo("adaptive", True),
                        "fixed_full_cost": mc("fixed", False), "fixed_prune_cost": mc("fixed", True),
                        "adapt_full_cost": mc("adaptive", False), "adapt_prune_cost": mc("adaptive", True),
                        "blend_save_pct": (1 - mc("adaptive", True) / base) * 100, "pareto_dom": dom})

    if args.out:
        pd.DataFrame(summary).to_parquet(args.out, index=False)
        print(f"\n[blend] wrote threshold-sweep summary -> {args.out}")


if __name__ == "__main__":
    main()
