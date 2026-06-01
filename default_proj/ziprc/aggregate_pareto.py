#!/usr/bin/env python3
"""Aggregate multiple per-seed Pareto summary JSONs into mean +/- std (error bars).

Each input JSON: {config: {acc, oracle, cost, latency}}. Writes a combined JSON with
mean/std per config per metric, prints a table, and saves an error-bar compute-Pareto
figure (cost vs acc with x/y std bars).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def aggregate(summaries):
    """summaries: list of {config: {metric: val}} -> {config: {metric: (mean,std,n)}}."""
    configs = sorted({k for s in summaries for k in s})
    metrics = ["acc", "oracle", "cost", "latency"]
    out = {}
    for cfg in configs:
        out[cfg] = {}
        for m in metrics:
            vals = [s[cfg][m] for s in summaries if cfg in s and m in s[cfg]]
            if vals:
                out[cfg][m] = [float(np.mean(vals)), float(np.std(vals)), len(vals)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Pareto summary JSONs (or globs).")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    summaries = [json.load(open(p)) for p in paths]
    print(f"[aggregate] {len(summaries)} seeds from {paths}")
    agg = aggregate(summaries)

    print(f"\n{'config':<12} {'acc (mean±std)':>16} {'cost':>14} {'latency':>14}")
    print("-" * 60)
    for cfg, m in agg.items():
        a = m.get("acc", [float('nan'), 0, 0]); c = m.get("cost", [float('nan'), 0, 0]); l = m.get("latency", [float('nan'), 0, 0])
        print(f"{cfg:<12} {a[0]:.3f}±{a[1]:.3f}      {c[0]:7.0f}±{c[1]:<5.0f} {l[0]:6.1f}±{l[1]:<4.1f}")

    if args.out_json:
        json.dump(agg, open(args.out_json, "w"), indent=2)
        print(f"[aggregate] wrote {args.out_json}")

    # error-bar compute Pareto (none + prune + utility)
    png = args.out_png or (args.out_json and args.out_json.replace(".json", ".png"))
    if png:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for cfg, m in agg.items():
            if "cost" not in m or "acc" not in m:
                continue
            if not (cfg in ("none", "prune") or cfg.startswith("util")):
                continue
            color = "#C44E52" if cfg == "none" else ("#DD8452" if cfg == "prune" else "#4C72B0")
            ax.errorbar(m["cost"][0], m["acc"][0], xerr=m["cost"][1], yerr=m["acc"][1],
                        fmt="o", color=color, capsize=3, zorder=3)
            ax.annotate(cfg, (m["cost"][0], m["acc"][0]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("cost (active forward passes)"); ax.set_ylabel("accuracy")
        ax.set_title("Compute–accuracy Pareto (multi-seed mean±std)")
        fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
        print(f"[aggregate] wrote {png}")


if __name__ == "__main__":
    main()
