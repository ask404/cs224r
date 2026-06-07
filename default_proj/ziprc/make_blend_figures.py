#!/usr/bin/env python3
"""Figures for the blend study:
  (A) Pareto frontier per pool — fixed+full across budgets vs the blend (adaptive+prune) cloud
      over budget x tau (matched-budget; from the frontier grids).
  (B) The calibration law — mid-trajectory separability (mid-AUC) vs the SIGNED prune accuracy-hit
      (from `blend_stats.py` law points: pool + per-tier), with Pearson and a PARTIAL correlation
      controlling base-oracle (difficulty). The signed y-axis (no 0-floor) and partial control are
      the fixes from the statistical-validity audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def pareto_front(points):
    pts = sorted(points, key=lambda x: x[0])
    front, best = [], -1.0
    for c, o in pts:
        if o > best + 1e-12:
            front.append((c, o))
            best = o
    return front


def _resid(y, x):
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _pear(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", required=True, help="name:frontier_grid.parquet per pool (panel A).")
    ap.add_argument("--law-points", nargs="*", default=None, help="blend_stats law_*.parquet (panel B).")
    ap.add_argument("--out", default="/vol/ziprc/figures/blend_study.png")
    args = ap.parse_args()

    pools = [(s.split(":", 1)[0], pd.read_parquet(s.split(":", 1)[1])) for s in args.sweeps]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.82, len(pools)))

    # (A) Pareto frontiers
    ax = axes[0]
    for (name, df), col in zip(pools, colors):
        ff = pareto_front(df.drop_duplicates("budget")[["fixed_full_cost", "fixed_full_orc"]].values.tolist())
        ax.plot([c for c, _ in ff], [o for _, o in ff], "o--", color=col, alpha=0.55, label=f"{name}: fixed+full")
        bl = pareto_front(list(zip(df["adapt_prune_cost"], df["adapt_prune_orc"])))
        ax.plot([c for c, _ in bl], [o for _, o in bl], "s-", color=col, lw=2.3, label=f"{name}: blend")
    ax.set_xlabel("compute (active forward passes / prompt)")
    ax.set_ylabel("oracle (pass@budget)")
    ax.set_title("(A) Pareto frontier — blend vs fixed, per pool")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (B) Calibration law from signed law points
    ax = axes[1]
    if args.law_points:
        law = pd.concat([pd.read_parquet(p) for p in args.law_points], ignore_index=True)
        law = law.dropna(subset=["mid_auc", "prune_hit", "base_oracle"])
        names = list(dict.fromkeys(law["name"]))
        cmap = {n: plt.cm.viridis(v) for n, v in zip(names, np.linspace(0, 0.82, len(names)))}
        for _, r in law.iterrows():
            big = r["level"] == "pool"
            ax.scatter(r["mid_auc"], r["prune_hit"], s=170 if big else 45,
                       c=[cmap[r["name"]]], edgecolors="k" if big else "none",
                       marker="o" if big else "^", zorder=3 if big else 2)
            if big:
                ax.annotate(r["name"], (r["mid_auc"], r["prune_hit"]), textcoords="offset points",
                            xytext=(8, 5), fontsize=9, fontweight="bold")
        x, y, z = law["mid_auc"].values, law["prune_hit"].values, law["base_oracle"].values
        if len(x) >= 4:
            pear = _pear(x, y)
            partial = _pear(_resid(x, z), _resid(y, z))
            b1, b0 = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, b0 + b1 * xx, "k--", alpha=0.5)
            ax.text(0.04, 0.06, f"Pearson r={pear:+.2f}\npartial r|base-oracle={partial:+.2f}  (n={len(x)})",
                    transform=ax.transAxes, fontsize=9, va="bottom",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("mid-trajectory separability  (mid-value AUC at prune point)")
        ax.set_ylabel("signed prune accuracy-hit  (oracle, ≤0; higher = safer)")
        ax.set_title("(B) Calibration law — separability predicts prune-safety")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
