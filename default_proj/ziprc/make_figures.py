#!/usr/bin/env python3
"""Generate report figures + results.md from ZIP-RC-Lite result artifacts.

Inputs (any subset):
  --scored        scored test parquet (prompt_idx, correct, value_first/_q25/_mean/_end,
                  response, target, nums) -> calibration + per-prefix AUC + selection curve
  --pareto        decode Pareto summary JSON ({config: {acc,oracle,cost,latency}})
                  -> cost-accuracy Pareto plot
  --struct-scored optional structured-head scored parquet -> structured-vs-binary calibration
Outputs: PNGs + results.md under --out-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from evaluation.countdown import evaluate_equation, extract_solution  # noqa: E402


def auc(labels, scores):
    labels, scores = np.asarray(labels, float), np.asarray(scores, float)
    npos, nneg = labels.sum(), (1 - labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def selection_curves(df, ks=(1, 2, 4, 8), seed=0):
    rng = np.random.default_rng(seed)
    out = {m: [] for m in ("random", "majority", "value", "oracle")}
    groups = [g for _, g in df.groupby("prompt_idx")]
    for K in ks:
        rnd, maj, val, orc = [], [], [], []
        for g in groups:
            if len(g) < K:
                continue
            sub = g.sample(n=K, random_state=int(rng.integers(1 << 30)))
            corr = sub["correct"].astype(float).values
            rnd.append(corr.mean()); orc.append(float(corr.max() >= 1.0))
            val.append(float(corr[int(np.argmax(sub["value_end"].values))]))
            tgt = int(sub["target"].iloc[0]); counts = {}
            for r in sub["response"].tolist():
                eq = extract_solution(r); v = evaluate_equation(eq) if eq else None
                if v is not None:
                    counts[round(float(v), 6)] = counts.get(round(float(v), 6), 0) + 1
            maj.append(float(abs(max(counts, key=counts.get) - tgt) < 1e-5) if counts else 0.0)
        out["random"].append(np.mean(rnd)); out["majority"].append(np.mean(maj))
        out["value"].append(np.mean(val)); out["oracle"].append(np.mean(orc))
    return {m: np.array(v) for m, v in out.items()}, list(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=None)
    ap.add_argument("--pareto", default=None)
    ap.add_argument("--adaptive-k", default=None)
    ap.add_argument("--struct-scored", default=None)
    ap.add_argument("--out-dir", default="/vol/ziprc/figures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    md = ["# ZIP-RC-Lite on Countdown — results\n"]

    if args.scored:
        df = pd.read_parquet(args.scored)
        df = df[df["value_end"].notna()].copy()
        # per-prefix AUC
        prefixes = [c for c in ("value_first", "value_q25", "value_mean", "value_end") if c in df.columns]
        aucs = {c: auc(df["correct"].values, df[c].values) for c in prefixes}
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(range(len(prefixes)), [aucs[c] for c in prefixes], color="#4C72B0")
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.set_xticks(range(len(prefixes))); ax.set_xticklabels([c.replace("value_", "") for c in prefixes])
        ax.set_ylabel("AUC(value→correct)"); ax.set_ylim(0.4, 1.0)
        ax.set_title("Introspective calibration by prefix position")
        fig.tight_layout(); fig.savefig(out / "calibration_auc.png", dpi=130); plt.close(fig)
        md.append("## Calibration (held-out)\n")
        md.append("| prefix | AUC(value→correct) |\n|---|---|\n" +
                  "".join(f"| {c.replace('value_','')} | {aucs[c]:.3f} |\n" for c in prefixes) + "\n")

        # selection curve
        curves, ks = selection_curves(df)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for m, c in [("oracle", "#999999"), ("majority", "#55A868"), ("value", "#4C72B0"), ("random", "#C44E52")]:
            ax.plot(ks, curves[m], marker="o", label=m, color=c)
        ax.set_xlabel("K (samples)"); ax.set_ylabel("accuracy"); ax.set_xticks(ks)
        ax.set_title("Selection: value vs majority vs oracle"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "selection_curve.png", dpi=130); plt.close(fig)
        md.append("## Selection @K (held-out)\n")
        md.append("| K | random | majority | value | oracle |\n|---|---|---|---|---|\n" +
                  "".join(f"| {k} | {curves['random'][i]:.3f} | {curves['majority'][i]:.3f} | "
                          f"{curves['value'][i]:.3f} | {curves['oracle'][i]:.3f} |\n"
                          for i, k in enumerate(ks)) + "\n")

    if args.pareto:
        summary = json.load(open(args.pareto))
        none = summary.get("none", {})

        def _frontier(xkey, keep, fname, xlabel, title, line_prefix):
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for k, v in summary.items():
                if not keep(k):
                    continue
                color = "#C44E52" if k == "none" else ("#DD8452" if k == "prune" else "#4C72B0")
                ax.scatter(v[xkey], v["acc"], color=color, zorder=3)
                ax.annotate(k, (v[xkey], v["acc"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
            line = sorted((v[xkey], v["acc"]) for k, v in summary.items() if k.startswith(line_prefix))
            if line:
                ax.plot([x for x, _ in line], [a for _, a in line], "-", color="#4C72B0", alpha=0.5, zorder=2)
            if none:
                ax.axhline(none.get("oracle", np.nan), ls="--", c="gray", lw=1, label="oracle@K")
            ax.set_xlabel(xlabel); ax.set_ylabel("accuracy"); ax.set_title(title); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(out / fname, dpi=130); plt.close(fig)

        # compute axis: none + prune + utility(beta)
        _frontier("cost", lambda k: k in ("none", "prune") or k.startswith("util"),
                  "pareto_compute.png", "cost (active forward passes)",
                  "Compute–accuracy (ZIP-RC utility/prune)", "util")
        # latency axis: none + prune + earlystop(tau)
        _frontier("latency", lambda k: k in ("none", "prune") or k.startswith("estop"),
                  "pareto_latency.png", "latency (decode steps)",
                  "Latency–accuracy (ZIP-RC earlystop)", "estop")
        md.append("## Cost–accuracy frontiers (held-out)\n")
        md.append("| config | acc | oracle | cost | latency |\n|---|---|---|---|---|\n" +
                  "".join(f"| {k} | {v['acc']:.3f} | {v['oracle']:.3f} | {v['cost']:.0f} | {v['latency']:.1f} |\n"
                          for k, v in summary.items()) + "\n")

    if args.adaptive_k:
        ak = json.load(open(args.adaptive_k))
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for s, c in [("fixed", "#C44E52"), ("hardness", "#4C72B0"), ("uncertainty", "#55A868")]:
            if s in ak:
                ax.plot(ak[s]["B"], ak[s]["oracle"], marker="o", label=s, color=c)
        ax.set_xlabel("mean K (budget)"); ax.set_ylabel("oracle accuracy")
        ax.set_title("Adaptive-K allocation by predicted hardness"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "adaptive_k.png", dpi=130); plt.close(fig)
        md.append("## Adaptive-K allocation (held-out, oracle@meanK)\n")
        md.append("| mean-K | fixed | hardness | uncertainty |\n|---|---|---|---|\n" +
                  "".join(f"| {b:g} | {ak['fixed']['oracle'][i]:.3f} | {ak['hardness']['oracle'][i]:.3f} | "
                          f"{ak['uncertainty']['oracle'][i]:.3f} |\n" for i, b in enumerate(ak["fixed"]["B"])) + "\n")

    (out / "results.md").write_text("".join(md))
    print(f"[figures] wrote {out}/results.md + PNGs")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
