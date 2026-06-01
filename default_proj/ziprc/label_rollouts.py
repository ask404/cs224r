#!/usr/bin/env python3
"""M2 + M3: label Countdown rollouts and audit bin occupancy.

Replaces the upstream 30B-235B grader: Countdown correctness is DETERMINISTIC
(compute_score), so we only spend a judge on the failure set to split coherent
vs incoherent.

Writes back two label columns onto the parquet:
  correct  in {0.0, 1.0}            -> binary baseline head
  reward3  in {0.0, 0.1, 1.0}       -> structured 3-outcome head

Also prints the M3 audit (the >=15%-per-class gate and the length-bin occupancy
that catches the degenerate-length-bin trap) BEFORE you spend a dollar training.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ziprc.config import (COHERENT, CORRECT, INCOHERENT, LENGTH_BINS_COUNTDOWN,
                          REWARD_VALUES_STRUCTURED)
from ziprc.grid import length_bin_of
from ziprc.verifier import binary_label, three_outcome_label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-parquet", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--judge", choices=["heuristic", "haiku"], default="heuristic",
                   help="Coherence backend for failures. 'heuristic' needs no API.")
    p.add_argument("--judge-model", default=None, help="Override ZIPRC_JUDGE_MODEL.")
    p.add_argument("--judge-cache", default=None, help="JSON cache path for Haiku labels.")
    p.add_argument("--min-class-frac", type=float, default=0.15,
                   help="M3 gate: warn if any structured class is below this fraction.")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.in_parquet)
    n = len(df)
    print(f"[label] {n} rollouts from {args.in_parquet} | judge={args.judge}", flush=True)

    judge = None
    if args.judge == "haiku":
        from ziprc.verifier import HaikuCoherenceJudge
        judge = HaikuCoherenceJudge(model=args.judge_model, cache_path=args.judge_cache)

    correct, reward3 = [], []
    for _, row in df.iterrows():
        gt = row["ground_truth"]
        resp = row["response"]
        correct.append(binary_label(resp, gt))
        reward3.append(three_outcome_label(resp, gt, judge=judge))
    if judge is not None:
        judge.flush()
        print(f"[label] Haiku API calls: {judge.n_api_calls} (failures only, cached)", flush=True)

    df["correct"] = correct
    df["reward3"] = reward3
    df.to_parquet(args.out_parquet, engine="pyarrow", index=False)
    print(f"[label] wrote {args.out_parquet}", flush=True)

    # ---- M3 audit: class occupancy (the >=15% gate) -----------------------
    c = Counter(reward3)
    print("\n=== M3 AUDIT: structured-class occupancy ===")
    names = {INCOHERENT: "incoherent", COHERENT: "coherent", CORRECT: "correct"}
    gate_ok = True
    for val in REWARD_VALUES_STRUCTURED:
        frac = c.get(val, 0) / n
        flag = "" if frac >= args.min_class_frac else "  <-- BELOW GATE"
        if frac < args.min_class_frac:
            gate_ok = False
        print(f"  {names[val]:<11} reward={val:<4} n={c.get(val,0):<5} ({frac:5.1%}){flag}")
    binary_acc = sum(correct) / n
    print(f"  [binary] correct rate: {binary_acc:.1%}")

    # ---- M3 audit: length-bin occupancy (degenerate-bin trap) -------------
    lengths = df["length"].tolist()
    lb_counts = Counter(length_bin_of(int(l), LENGTH_BINS_COUNTDOWN) for l in lengths)
    print("\n=== M3 AUDIT: response-length-bin occupancy ===")
    L = len(LENGTH_BINS_COUNTDOWN) - 1
    bin0_frac = lb_counts.get(0, 0) / n
    for b in range(L):
        lo, hi = LENGTH_BINS_COUNTDOWN[b], LENGTH_BINS_COUNTDOWN[b + 1]
        frac = lb_counts.get(b, 0) / n
        print(f"  bin{b} [{lo:>4},{hi:>4}) n={lb_counts.get(b,0):<5} ({frac:5.1%})")
    if bin0_frac > 0.9:
        print("  WARNING: >90% of responses in length bin 0 -> length signal will be "
              "degenerate. Shrink LENGTH_BINS_COUNTDOWN in ziprc/config.py.")

    print("\n=== GATE SUMMARY ===")
    struct_msg = "PASS" if gate_ok else "FAIL (fall back to binary, report as result)"
    length_msg = "PASS" if bin0_frac <= 0.9 else "FAIL (retune bins)"
    print(f"  structured >=15% per class: {struct_msg}")
    print(f"  length bins non-degenerate: {length_msg}")


if __name__ == "__main__":
    main()
