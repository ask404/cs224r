#!/usr/bin/env python3
"""Cross-pool calibration-law test. Ingests per-tier + pool law points from `blend_stats.py`
(columns: name, level, tier, n, mid_auc, prune_hit, base_oracle) and asks, rigorously:

  Does mid-trajectory separability (mid_auc) predict prune's SIGNED accuracy hit (prune_hit, <=0,
  closer to 0 = safer) ABOVE AND BEYOND difficulty (proxied by base_oracle)?

Reports Pearson/Spearman, a PERMUTATION p-value (shuffle mid_auc), and the PARTIAL correlation
of (mid_auc, prune_hit | base_oracle) with its own permutation p-value. If the partial correlation
survives, the law is not merely the difficulty gradient (the audit's central confound).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def _resid(y, x):
    """Residual of y after OLS on [1, x]."""
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _perm_p(a, b, stat_fn, B=20000, seed=0):
    """Two-sided permutation p-value for |stat| by shuffling `a`."""
    obs = abs(stat_fn(a, b))
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        if abs(stat_fn(rng.permutation(a), b)) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (B + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law-points", nargs="+", required=True, help="blend_stats --out-law parquets.")
    ap.add_argument("--use", choices=["all", "tier", "pool"], default="all",
                    help="Which points to correlate (all=pool+tier).")
    ap.add_argument("--min-n", type=int, default=10, help="Drop points with fewer prompts.")
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in args.law_points], ignore_index=True)
    if args.use != "all":
        df = df[df["level"] == args.use]
    df = df[df["n"] >= args.min_n].dropna(subset=["mid_auc", "prune_hit", "base_oracle"]).reset_index(drop=True)
    x = df["mid_auc"].values
    y = df["prune_hit"].values            # signed, <= 0 ; higher mid_auc -> hit closer to 0 -> positive r
    z = df["base_oracle"].values

    print(f"\n################ CALIBRATION LAW (n={len(df)} points: "
          f"{(df['level'] == 'pool').sum()} pool + {(df['level'] == 'tier').sum()} tier) ################")
    print("  x = mid-trajectory separability (mid-AUC) ; y = SIGNED prune accuracy-hit (<=0, higher=safer)")
    for _, r in df.iterrows():
        print(f"    {r['name']:>10} {r['level']:>4} t{int(r['tier']):>2}  n={int(r['n']):>3}  "
              f"mid-AUC={r['mid_auc']:.3f}  hit={r['prune_hit']:+.3f}  base-orc={r['base_oracle']:.3f}")

    if len(df) < 4:
        print("\n  too few points for inference.")
        return

    pear = _pearson(x, y)
    rho = _pearson(np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float))
    p_perm = _perm_p(x, y, _pearson)
    print(f"\n  RAW:     Pearson r={pear:+.2f}  Spearman rho={rho:+.2f}  permutation p={p_perm:.4f}")

    # partial correlation controlling for base_oracle (difficulty proxy)
    rx, ry = _resid(x, z), _resid(y, z)
    partial = _pearson(rx, ry)
    p_partial = _perm_p(rx, ry, _pearson)
    # how much does base_oracle alone explain each axis?
    print(f"  base-oracle vs mid-AUC r={_pearson(z, x):+.2f} ; base-oracle vs hit r={_pearson(z, y):+.2f}")
    print(f"  PARTIAL (mid-AUC, hit | base-oracle): r={partial:+.2f}  permutation p={p_partial:.4f}")
    verdict = ("LAW SURVIVES difficulty-control" if (p_partial < 0.05 and partial > 0)
               else "NOT separable from difficulty" if not np.isnan(partial)
               else "inconclusive")
    print(f"  >>> {verdict}  (partial r>0 & p<0.05 => separability predicts prune-safety beyond difficulty)")


if __name__ == "__main__":
    main()
