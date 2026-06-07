#!/usr/bin/env python3
"""Mid-trajectory adaptive-K: allocate the sample budget from a SHORT PROBE, not from the
prompt (which fails OOD — see ADAPTIVE_K.md / the difficulty experiment: value_first AUC
0.24 vs value_q25 AUC 0.84 on hard Countdown).

Two stages, at matched average budget B:
  1. PROBE: give every prompt `probe_k` samples; read each sample's `--signal-col` (default
     value_q25 ~ "is this attempt going well after ~25% of generation"); a prompt is
     `probe_solved` if any probe sample is correct (verifiable task -> we know).
  2. REALLOCATE: don't spend more on probe-solved prompts; pour the freed budget into the
     UNSOLVED prompts, weighted by the probe signal:
       - frontier: w ∝ promise·(1−promise)   (medium promise = where extra samples flip)
       - promise : w ∝ promise               (chase the most-promising unsolved)
Compare oracle@B to fixed (uniform) allocation. Fully offline; reuses the scored rollouts.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def two_stage(groups, B, kmax, probe_k, scheme, rng):
    """Return mean oracle over prompts under a matched total budget B*n."""
    n = len(groups)
    total = int(round(B * n))
    # --- stage 1: probe ---
    promise, solved, pools = [], [], []
    base_cost = 0
    for g in groups:
        idx = rng.permutation(len(g))
        probe = g.iloc[idx[:probe_k]]
        rest = g.iloc[idx[probe_k:]]
        promise.append(float(probe["__sig"].mean()))
        s = bool(probe["correct"].max() >= 1.0)
        solved.append(s)
        pools.append(rest["correct"].astype(float).values)  # remaining samples to draw from
        base_cost += probe_k
    promise = np.array(promise); solved = np.array(solved)
    # rank-normalize promise (absolute values are compressed; ordering is what carries)
    pr = np.empty(n); pr[np.argsort(promise)] = np.linspace(0, 1, n)

    # --- stage 2: distribute the remaining budget ---
    extra_total = max(0, total - base_cost)
    K_extra = np.zeros(n, dtype=int)
    cap = kmax - probe_k
    if scheme == "fixed":
        e = extra_total // n
        K_extra[:] = e
        K_extra[: extra_total - e * n] += 1
        K_extra = np.minimum(K_extra, cap)
    else:
        if scheme == "frontier":
            w = pr * (1 - pr)
        elif scheme == "promise":
            w = pr.copy()
        else:
            raise ValueError(scheme)
        w = np.where(solved, 0.0, w + 1e-6)   # nothing extra for already-solved prompts
        rem = extra_total
        while rem > 0 and (w > 0).any():
            avail = (K_extra < cap) & (~solved) & (w > 0)
            if not avail.any():
                break
            pri = np.where(avail, w / (K_extra + 1), -np.inf)
            K_extra[int(np.argmax(pri))] += 1
            rem -= 1

    # --- evaluate oracle ---
    orc = []
    for i, g in enumerate(groups):
        if solved[i]:
            orc.append(1.0)
            continue
        k = min(K_extra[i], len(pools[i]))
        orc.append(float(pools[i][:k].max() >= 1.0) if k > 0 else 0.0)
    return float(np.mean(orc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--signal-col", default="value_q25")
    ap.add_argument("--probe-k", type=int, default=2)
    ap.add_argument("--budgets", type=float, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.scored)
    df = df[df[args.signal_col].notna()].copy()
    df["__sig"] = df[args.signal_col]
    groups = [g.reset_index(drop=True) for _, g in df.groupby("prompt_idx") if len(g) >= args.kmax]
    print(f"[mid-K] {len(groups)} prompts | signal={args.signal_col} probe_k={args.probe_k} kmax={args.kmax}")

    schemes = ["fixed", "frontier", "promise"]
    res = {s: {"B": [], "oracle": []} for s in schemes}
    hdr = f"{'meanK':>6} | " + " | ".join(f"{s:>9}" for s in schemes) + " |  frontier−fixed"
    print("\n" + hdr); print("-" * len(hdr))
    for B in args.budgets:
        row, vals = f"{B:6.1f} | ", {}
        for s in schemes:
            o = float(np.mean([two_stage(groups, B, args.kmax, args.probe_k, s, np.random.default_rng(args.seed + t))
                               for t in range(args.trials)]))
            vals[s] = o; res[s]["B"].append(B); res[s]["oracle"].append(o)
            row += f" {o:9.3f}"
        row += f" |  {vals['frontier'] - vals['fixed']:+.3f}"
        print(row)

    print("\n[mid-K] frontier gain over fixed (= does a MID-TRAJECTORY probe fix adaptive-K?):")
    for i, B in enumerate(args.budgets):
        print(f"  meanK={B:.1f}: frontier {res['frontier']['oracle'][i]-res['fixed']['oracle'][i]:+.3f} | "
              f"promise {res['promise']['oracle'][i]-res['fixed']['oracle'][i]:+.3f}")
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=2)
        print(f"[mid-K] wrote {args.out_json}")


if __name__ == "__main__":
    main()
