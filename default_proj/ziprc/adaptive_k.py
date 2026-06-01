#!/usr/bin/env python3
"""Stage C2 (adaptive allocation): spend more samples on prompts the head predicts
are HARD, fewer on easy ones — branching's "reallocate compute" idea without any
KV surgery.

The head's `value_first` (its value prediction at the prompt, before any sampling)
is a per-prompt difficulty estimate (AUC ~0.85 vs correctness). Given a fixed average
sample budget B, we allocate K_i samples per prompt by a difficulty weight, then ask:
does adaptive allocation beat fixed-K at the SAME average cost?

Fully offline: reuses a scored parquet (prompt_idx, correct, value_first, value_end),
so no new generation. Compares, per budget B:
  fixed        : every prompt gets B samples
  uncertainty  : K_i ~ d_i(1-d_i)   (most samples to medium-difficulty, where extra
                 tries flip the outcome)
  hardness     : K_i ~ (1 - d_i)     (most samples to predicted-hard prompts)
Metric: oracle (any of K_i correct) and value-selection accuracy, at matched mean-K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def allocate(d, B, kmax, scheme, rng):
    n = len(d)
    T = int(max(n, min(kmax * n, round(B * n))))
    K = np.ones(n, dtype=int)
    if scheme == "fixed":
        base, rem = divmod(T, n)
        K[:] = base
        idx = rng.permutation(n)[:rem]
        K[idx] += 1
        return np.clip(K, 1, kmax)
    if scheme == "uncertainty":
        w = d * (1 - d) + 1e-3
    elif scheme == "hardness":
        w = (1 - d) + 1e-3
    else:
        raise ValueError(scheme)
    rem = T - n
    while rem > 0:
        avail = K < kmax
        if not avail.any():
            break
        pri = np.where(avail, w / K, -np.inf)   # high weight, low current K first
        K[int(np.argmax(pri))] += 1
        rem -= 1
    return K


def evaluate(groups, d, B, kmax, scheme, rng, trials=12):
    orc, vsel = [], []
    for _ in range(trials):
        K = allocate(d, B, kmax, scheme, rng)
        o, v = [], []
        for i, g in enumerate(groups):
            sub = g.sample(n=int(K[i]), random_state=int(rng.integers(1 << 30)))
            corr = sub["correct"].astype(float).values
            o.append(float(corr.max() >= 1.0))
            v.append(float(corr[int(np.argmax(sub["value_end"].values))]))
        orc.append(np.mean(o)); vsel.append(np.mean(v))
    return float(np.mean(orc)), float(np.mean(vsel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--budgets", type=float, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.scored)
    df = df[df["value_end"].notna() & df["value_first"].notna()].copy()
    groups = [g.reset_index(drop=True) for _, g in df.groupby("prompt_idx") if len(g) >= args.kmax]
    d_raw = np.array([g["value_first"].mean() for g in groups])
    # value_first RANKS difficulty well (AUC ~0.85) but its absolute values are compressed,
    # so allocate by PERCENTILE RANK (full [0,1] spread), not the raw value.
    d = np.empty_like(d_raw)
    d[d_raw.argsort()] = np.linspace(0.0, 1.0, len(d_raw))  # 0 = hardest, 1 = easiest
    print(f"[adaptive_k] {len(groups)} prompts; value_first raw spread "
          f"[{d_raw.min():.2f},{d_raw.max():.2f}] -> rank-normalized to [0,1] for allocation")

    schemes = ["fixed", "uncertainty", "hardness"]
    results = {s: {"B": [], "oracle": [], "value": []} for s in schemes}
    hdr = f"{'meanK':>6} | " + " | ".join(f"{s+' orc/val':>16}" for s in schemes)
    print("\n" + hdr); print("-" * len(hdr))
    for B in args.budgets:
        row = f"{B:6.1f} | "
        for s in schemes:
            o, v = evaluate(groups, d, B, args.kmax, s, np.random.default_rng(args.seed), args.trials)
            results[s]["B"].append(B); results[s]["oracle"].append(o); results[s]["value"].append(v)
            row += f"  {o:.3f}/{v:.3f}     "
        print(row)

    # headline: adaptive vs fixed at matched budget (oracle gain)
    print("\n[adaptive_k] oracle gain over fixed-K at matched mean-K:")
    for i, B in enumerate(args.budgets):
        fu = results["uncertainty"]["oracle"][i] - results["fixed"]["oracle"][i]
        fh = results["hardness"]["oracle"][i] - results["fixed"]["oracle"][i]
        print(f"  meanK={B:.1f}: uncertainty {fu:+.3f}, hardness {fh:+.3f}")

    if args.out_json:
        json.dump(results, open(args.out_json, "w"), indent=2)
        print(f"[adaptive_k] wrote {args.out_json}")


if __name__ == "__main__":
    main()
