#!/usr/bin/env python3
"""Build a leakage-safe held-out evaluation pool from a train slice that is DISJOINT, by
problem identity (target, sorted nums), from the prompts the head actually trained on.

The main head trained on train[0:512]'s rollouts. A high-index train slice is index-disjoint,
but Countdown problems can recur across indices, so we additionally drop any candidate whose
(target, sorted nums) key appears in the head's training set. Output is head-clean; it remains
policy-seen (the frozen policy RL-trained on the train split), which we report transparently —
the blend/calibration claims are about the HEAD + test-time mechanism, for which head-disjoint
is the requirement.
"""
from __future__ import annotations

import argparse

import pandas as pd
from datasets import load_dataset


def _key(target, nums):
    return (int(target), tuple(sorted(int(x) for x in nums)))


def _tn(row, cols):
    """(target, nums) from a row that has either target/nums columns or a ground_truth dict."""
    if "target" in cols and "nums" in cols:
        return row["target"], row["nums"]
    g = row["ground_truth"]
    return g["target"], (g["numbers"] if "numbers" in g else g["nums"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="asingh15/countdown_tasks_3to4")
    ap.add_argument("--split", default="train")
    ap.add_argument("--offset", type=int, default=200000)
    ap.add_argument("--n-candidates", type=int, default=1200)
    ap.add_argument("--n-keep", type=int, default=300)
    ap.add_argument("--exclude-parquet", required=True,
                    help="The head's TRAINING data (e.g. train_labeled_512.parquet) -> excluded keys.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ex = pd.read_parquet(args.exclude_parquet)
    exc = set()
    for _, r in ex.iterrows():
        t, nums = _tn(r, ex.columns)
        exc.add(_key(t, nums))
    print(f"[holdout] head-training keys to exclude: {len(exc)} (from {args.exclude_parquet})", flush=True)

    ds = load_dataset(args.dataset, split=args.split)
    end = min(args.offset + args.n_candidates, len(ds))
    ds = ds.select(range(args.offset, end))
    cols = ds.column_names

    rows, seen = [], set()
    n_train_overlap, n_dup = 0, 0
    for i in range(len(ds)):
        row = ds[i]
        t, nums = _tn(row, cols)
        k = _key(t, nums)
        if k in exc:
            n_train_overlap += 1
            continue
        if k in seen:
            n_dup += 1
            continue
        seen.add(k)
        gt = row["ground_truth"] if "ground_truth" in cols else {"target": int(t), "numbers": [int(x) for x in nums]}
        rows.append({"prompt_idx": args.offset + i, "prompt": row["prompt"],
                     "target": int(t), "nums": [int(x) for x in nums], "ground_truth": gt})
        if len(rows) >= args.n_keep:
            break

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"[holdout] candidates scanned={i + 1} | excluded(train-overlap)={n_train_overlap} "
          f"excluded(dup)={n_dup} | kept={len(out)} -> {args.out}", flush=True)
    # belt-and-suspenders assertion: zero overlap with the exclusion set
    kept_keys = {_key(r["target"], r["nums"]) for _, r in out.iterrows()}
    assert kept_keys.isdisjoint(exc), "LEAKAGE: kept prompts overlap head-training problems!"
    print(f"[holdout] LEAKAGE CHECK PASSED: {len(kept_keys)} kept keys disjoint from head-training set.", flush=True)


if __name__ == "__main__":
    main()
