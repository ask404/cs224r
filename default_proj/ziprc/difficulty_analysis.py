#!/usr/bin/env python3
"""Difficulty-stratified analysis for the harder-Countdown experiment (EXPERIMENT_difficulty.md).

Reads a scored rollout parquet (correct, value_first, nums) and reports the two diagnostics
that decide whether adaptive-K should work:
  1. the dynamic RANGE of value_first (vs Countdown-3to4's dead [0.52, 0.54]); and
  2. per-difficulty-tier (= len(nums)) pass@1 and mean value_first — i.e. does the head's
     difficulty signal track TRUE difficulty, and is there real spread to reallocate over.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.scored)
    df = df[df["value_first"].notna()].copy()
    df["tier"] = df["nums"].apply(len)  # operand cardinality = difficulty

    vf = df.groupby("prompt_idx")["value_first"].mean().values
    rng = {"min": float(vf.min()), "median": float(np.median(vf)), "max": float(vf.max()),
           "std": float(vf.std()), "spread": float(vf.max() - vf.min())}
    print("=== value_first dynamic range (per prompt) ===")
    print(f"  min {rng['min']:.3f} | median {rng['median']:.3f} | max {rng['max']:.3f} | "
          f"std {rng['std']:.3f} | spread {rng['spread']:.3f}")
    print("  (Countdown-3to4 reference: spread ~0.02 -> adaptive-K was null)")

    print("\n=== per-difficulty-tier (n_numbers) ===")
    print(f"  {'tier':>4} {'prompts':>8} {'samples':>8} {'pass@1':>8} {'mean value_first':>17}")
    tiers = {}
    for t, g in df.groupby("tier"):
        n_prompts = g["prompt_idx"].nunique()
        p1 = float(g["correct"].mean())
        mvf = float(g["value_first"].mean())
        tiers[int(t)] = {"prompts": int(n_prompts), "samples": int(len(g)), "pass@1": p1, "value_first": mvf}
        print(f"  {t:>4} {n_prompts:>8} {len(g):>8} {p1:>8.3f} {mvf:>17.3f}")

    # Does the signal track true difficulty? (corr of per-prompt value_first with -tier)
    per_prompt = df.groupby("prompt_idx").agg(vf=("value_first", "mean"), tier=("tier", "first"))
    corr = float(np.corrcoef(per_prompt["vf"], -per_prompt["tier"])[0, 1]) if len(per_prompt) > 1 else float("nan")
    print(f"\n  corr(value_first, EASINESS=-tier) = {corr:.3f}  "
          f"(positive => head's signal tracks true difficulty)")

    res = {"value_first_range": rng, "tiers": tiers, "corr_value_first_easiness": corr}
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=2)
        print(f"\n[difficulty] wrote {args.out_json}")


if __name__ == "__main__":
    main()
