#!/usr/bin/env python3
"""Figures for the blend study:
  (A) Pareto frontier per pool — fixed+full across budgets vs the blend (adaptive+prune)
      cloud across budget x tau; the blend frontier should sit up-and-left (dominates).
  (B) The calibration law — mid-trajectory winner/loser separability (mid-AUC) vs the blend's
      best Pareto compute-saving at >= baseline oracle, one point per pool.

Input: one or more `name:parquet:midauc` specs (midauc scraped from each run's CALIBRATION
DIAGNOSTIC line). Each parquet is a blend_eval frontier grid (budget,tau,*_orc,*_cost,...).
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def pareto_front(points):
    """Lower-left-favoring upper frontier: max oracle at each cost. points: list of (cost, orc)."""
    pts = sorted(points, key=lambda x: x[0])
    front, best = [], -1.0
    for c, o in pts:
        if o > best + 1e-12:
            front.append((c, o))
            best = o
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", required=True, help="name:parquet:midauc per pool.")
    ap.add_argument("--out", default="/vol/ziprc/figures/blend_study.png")
    args = ap.parse_args()

    pools = []
    for spec in args.sweeps:
        name, path, mid = spec.split(":")
        pools.append((name, pd.read_parquet(path), float(mid)))

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (A) Pareto frontiers
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(pools)))
    for (name, df, mid), col in zip(pools, colors):
        # fixed+full frontier: one (cost, orc) per budget (prune-independent -> dedup by budget)
        ff = df.drop_duplicates("budget")[["fixed_full_cost", "fixed_full_orc"]].values.tolist()
        ff = pareto_front([(c, o) for c, o in ff])
        ax.plot([c for c, _ in ff], [o for _, o in ff], "o--", color=col, alpha=0.5,
                label=f"{name}: fixed+full")
        # blend frontier: best oracle at each cost over ALL (budget,tau) adapt+prune points
        bl = pareto_front(list(zip(df["adapt_prune_cost"], df["adapt_prune_orc"])))
        ax.plot([c for c, _ in bl], [o for _, o in bl], "s-", color=col, linewidth=2.2,
                label=f"{name}: blend (adapt+prune)")
    ax.set_xlabel("compute (active forward passes / prompt)")
    ax.set_ylabel("oracle (pass@budget)")
    ax.set_title("(A) Pareto frontier: blend dominates where the head is calibrated")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (B) Calibration law
    ax = axes[1]
    xs, ys, names = [], [], []
    for name, df, mid in pools:
        # best Pareto compute-saving among blend points with oracle >= that pool's max fixed+full oracle
        base = df["fixed_full_orc"].max()
        ok = df[df["adapt_prune_orc"] >= base - 1e-9]
        save = (1 - ok["adapt_prune_cost"].min() / df.loc[df["fixed_full_orc"].idxmax(), "fixed_full_cost"]) * 100 if len(ok) else 0.0
        xs.append(mid)
        ys.append(max(0.0, save))
        names.append(name)
    order = np.argsort(xs)
    xs, ys = np.array(xs), np.array(ys)
    ax.plot(xs[order], ys[order], "k--", alpha=0.4)
    ax.scatter(xs, ys, s=140, c=plt.cm.viridis(np.linspace(0, 0.85, len(pools))), zorder=3)
    for x, y, nm in zip(xs, ys, names):
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("mid-trajectory winner/loser separability  (mid-value AUC at prune point)")
    ax.set_ylabel("blend Pareto compute-saving at ≥ baseline oracle  (%)")
    ax.set_title("(B) The calibration law: separability predicts the free lunch")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"[fig] wrote {args.out}")
    for name, df, mid in pools:
        base = df["fixed_full_orc"].max()
        ok = df[df["adapt_prune_orc"] >= base - 1e-9]
        print(f"  {name:10s} mid-AUC={mid:.3f}  base-oracle={base:.3f}  "
              f"pareto-points={len(ok)}/{len(df)}")


if __name__ == "__main__":
    main()
