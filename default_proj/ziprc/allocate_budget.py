#!/usr/bin/env python3
"""Online probe-and-reallocate, stage 2: from a scored PROBE, decide how many FRESH extra
samples to generate per prompt. Skip prompts the probe already solved; pour the budget into
unsolved prompts weighted by the mid-trajectory signal (value_q25). Writes one row per prompt
(prompt_idx, prompt, target, nums, ground_truth, n_extra) for `gen_rollouts --n-col n_extra`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="Scored rollouts; first --probe-k per prompt are the probe.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, required=True, help="Target mean total samples/prompt (probe+extra).")
    ap.add_argument("--probe-k", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--signal-col", default="value_q25")
    ap.add_argument("--scheme", choices=["frontier", "promise"], default="frontier")
    args = ap.parse_args()

    df = pd.read_parquet(args.scored)
    promise, solved, meta = [], [], []
    for pi, g in df.groupby("prompt_idx"):
        g = g.reset_index(drop=True)
        probe = g.iloc[: args.probe_k]
        promise.append(float(probe[args.signal_col].mean()))
        solved.append(bool(probe["correct"].max() >= 1.0))
        r = g.iloc[0]
        meta.append((int(pi), r["prompt"], int(r["target"]), list(r["nums"]), r["ground_truth"]))

    n = len(meta)
    promise = np.array(promise); solved = np.array(solved)
    pr = np.empty(n); pr[np.argsort(promise)] = np.linspace(0, 1, n)  # rank-normalize the signal
    w = pr * (1 - pr) if args.scheme == "frontier" else pr.copy()
    w = np.where(solved, 0.0, w + 1e-6)                                # nothing for already-solved
    extra_total = max(0, int(round((args.budget - args.probe_k) * n)))
    cap = args.kmax - args.probe_k
    K = np.zeros(n, dtype=int); rem = extra_total
    while rem > 0 and (w > 0).any():
        avail = (K < cap) & (~solved) & (w > 0)
        if not avail.any():
            break
        K[int(np.argmax(np.where(avail, w / (K + 1), -np.inf)))] += 1
        rem -= 1

    rows = [{"prompt_idx": pi, "prompt": pr_, "target": t, "nums": nm, "ground_truth": gt, "n_extra": int(K[i])}
            for i, (pi, pr_, t, nm, gt) in enumerate(meta)]
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"[allocate] budget={args.budget} probe_k={args.probe_k} | {n} prompts | "
          f"{int(solved.sum())} probe-solved (skip) | extra allocated={int(K.sum())} "
          f"(mean total {args.probe_k + K.mean():.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
