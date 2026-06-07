#!/usr/bin/env python3
"""Train/test leakage audit. A Countdown problem is uniquely keyed by
(target, sorted(numbers)); we check that no problem in set A also appears in set B,
and report intra-set duplicates. Works on any parquet with `target` + `nums` columns
(problem sets or rollout sets — rollouts dedup to their underlying problems)."""
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd


def keys(path):
    df = pd.read_parquet(path)
    tcol = "target" if "target" in df.columns else None
    ncol = "nums" if "nums" in df.columns else None
    if tcol is None or ncol is None:  # fall back to ground_truth dicts
        ks = [(int(g["target"]), tuple(sorted(int(x) for x in g["numbers"]))) for g in df["ground_truth"]]
    else:
        ks = [(int(t), tuple(sorted(int(x) for x in n))) for t, n in zip(df[tcol], df[ncol])]
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ka, kb = keys(args.a), keys(args.b)
    sa, sb = set(ka), set(kb)
    overlap = sa & sb
    dup_a = sum(c - 1 for c in Counter(ka).values() if c > 1)
    dup_b = sum(c - 1 for c in Counter(kb).values() if c > 1)

    print(f"=== leakage check {args.label} ===")
    print(f"  A ({args.a.split('/')[-1]}): {len(ka)} rows, {len(sa)} unique problems, {dup_a} intra dups")
    print(f"  B ({args.b.split('/')[-1]}): {len(kb)} rows, {len(sb)} unique problems, {dup_b} intra dups")
    print(f"  >>> A∩B overlap: {len(overlap)} problems "
          f"({100*len(overlap)/max(1,len(sb)):.1f}% of B's unique)  "
          f"{'LEAKAGE!' if overlap else 'CLEAN ✓'}")
    for k in list(overlap)[:5]:
        print(f"      leaked: target={k[0]} nums={list(k[1])}")


if __name__ == "__main__":
    main()
