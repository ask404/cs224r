#!/usr/bin/env python3
"""M6/M7 KEYSTONE: is the value head good enough to be worth a live decoder?

Necessary condition for ZIP-RC adaptive sampling: the head's predicted value must
RANK completed rollouts (argmax-value selection beats random, approaches oracle).
This is offline and ~free. If it fails, do NOT build the live decoder — report
calibration + structured-vs-binary and stop.

Consumes a scored parquet (prompt_idx, correct, value_end, response, target, nums)
and reports, per K in --ks:
  random@K  : pick a sample uniformly         (= mean correctness; chance floor)
  major@K   : self-consistency plurality vote on the evaluated answer value
  value@K   : pick argmax predicted value_end (ZIP-RC selection)  <-- the result
  oracle@K  : 1 if any of the K is correct    (ceiling)
plus rank quality of value_end vs correctness: Pearson corr and ROC-AUC.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.countdown import evaluate_equation, extract_solution  # noqa: E402


def auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def answer_value(response: str):
    eq = extract_solution(response)
    return None if eq is None else evaluate_equation(eq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--value-col", default="value_end")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.in_parquet)
    df = df[df[args.value_col].notna()].copy()
    groups = list(df.groupby("prompt_idx"))
    print(f"[select] {len(groups)} prompts | value-col={args.value_col}", flush=True)

    # global ranking quality
    print(f"[select] corr(value, correct) = {np.corrcoef(df['correct'], df[args.value_col])[0,1]:.3f}")
    print(f"[select] ROC-AUC(value -> correct) = {auc(df['correct'].values, df[args.value_col].values):.3f}")

    # EARLY-vs-LATE: value_first/_q25 test mid-gen pruning viability; value_end is the
    # easy selection case (answer already written). High value_first AUC => live decoder viable.
    print("[select] ranking quality by prefix position (AUC, value -> correct):")
    for col in ["value_first", "value_q25", "value_mean", "value_end"]:
        if col in df.columns and df[col].notna().any():
            m = df[col].notna()
            print(f"    {col:<12} AUC={auc(df.loc[m, 'correct'].values, df.loc[m, col].values):.3f}")

    header = f"{'K':>3} | {'random':>7} {'major':>7} {'value':>7} {'oracle':>7} | value-random  oracle-gap"
    print("\n" + header)
    print("-" * len(header))
    for K in args.ks:
        rnd, maj, val, orc = [], [], [], []
        for _, g in groups:
            if len(g) < K:
                continue
            sub = g.sample(n=K, random_state=int(rng.integers(1 << 30)))
            corr = sub["correct"].astype(float).values
            vals = sub[args.value_col].values
            rnd.append(corr.mean())
            orc.append(float(corr.max() >= 1.0))
            val.append(float(corr[int(np.argmax(vals))]))
            # self-consistency: plurality over evaluated answer values
            answers = [answer_value(r) for r in sub["response"].tolist()]
            tgt = int(sub["target"].iloc[0])
            counts = {}
            for a in answers:
                if a is not None:
                    counts[round(float(a), 6)] = counts.get(round(float(a), 6), 0) + 1
            if counts:
                plurality = max(counts, key=counts.get)
                maj.append(float(abs(plurality - tgt) < 1e-5))
            else:
                maj.append(0.0)
        if not val:
            print(f"{K:>3} | (no groups with >= {K} samples)")
            continue
        r, m, v, o = np.mean(rnd), np.mean(maj), np.mean(val), np.mean(orc)
        print(f"{K:>3} | {r:7.3f} {m:7.3f} {v:7.3f} {o:7.3f} | "
              f"{v-r:+11.3f}  {o-v:9.3f}")

    print("\n[select] VIABILITY READ:")
    print("  value@K > random@K (and trending toward oracle@K)  => head ranks rollouts; "
          "live decoder is worth building.")
    print("  value@K ~= random@K                                => head can't rank; stop at "
          "calibration + structured-vs-binary.")


if __name__ == "__main__":
    main()
