#!/usr/bin/env python3
"""Rigorous offline statistics for the blend study, from a raw per-sample dump
(`blend_eval.py --dump-samples`). Generation is done; everything here is CPU + reproducible.

Addresses the statistical-validity audit:
  * PROMPT is the inference unit -> all CIs are prompt-level paired bootstraps (not seed-std,
    which is only generation noise). CIs are explicitly scoped "on these prompts".
  * HELD-OUT operating point: (budget, tau) is selected on one half of prompts and the blend's
    saving + oracle-delta are REPORTED on the other half (2-fold) -> kills winner's-curse / forking
    paths. The full grid is also printed.
  * MATCHED-BUDGET comparison (blend at B vs fixed+full at the SAME B) -> no cross-budget bug.
  * Falsifiable forms: COMPOUNDING = is blend saving == product of the two levers' savings (CI on
    the gap)? ALLOCATION-AGNOSTIC = is prune's token-fraction equal under fixed vs adaptive (CI)?
    SYNERGY = DiD of oracle (CI). CALIBRATION LAW = SIGNED prune accuracy-hit vs mid-AUC, per tier,
    fed to a cross-pool partial-correlation + permutation test (`law_combine.py`).
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
from ziprc.allocate_budget import allocate  # noqa: E402
from ziprc.blend_core import _auc, _smooth, sim  # noqa: E402

ARMS = ("fixed", "adaptive")


def reconstruct(dump, probe_k, kmax, scheme, budgets, taus, warmup, interval, keep_min, w):
    """Rebuild pools per seed, allocate online per budget, and compute per-prompt (cost, solved)
    for every (budget, tau, arm, prune). Average over seeds -> per-prompt estimates."""
    seeds = sorted(dump["seed"].unique())
    pidxs = sorted(dump["prompt_idx"].unique())
    acc = {(b, t, a, pr): {pi: {"c": [], "s": []} for pi in pidxs}
           for b in budgets for t in taus for a in ARMS for pr in (False, True)}
    tier_of, diag = {}, []
    for seed in seeds:
        sd = dump[dump["seed"] == seed]
        pools = {}
        for pi, g in sd.groupby("prompt_idx"):
            g = g.sort_values("sample")
            pool = [{"vhist": list(v), "n_tokens": int(n), "correct": bool(c), "q25": float(q)}
                    for v, n, c, q in zip(g["vhist"], g["n_tokens"], g["correct"], g["q25"])]
            pools[pi] = pool
            tier_of[pi] = int(g["tier"].iloc[0])
            for sm in pool:
                vh = sm["vhist"]
                reached = sm["n_tokens"] > warmup
                diag.append({"tier": tier_of[pi], "correct": int(sm["correct"]), "reached": reached,
                             "mid": _smooth(vh[: warmup + 1], w) if reached else np.nan})
        order = sorted(pools)
        promise = np.array([np.mean([pools[pi][k]["q25"] for k in range(probe_k)]) for pi in order])
        solved = np.array([any(pools[pi][k]["correct"] for k in range(probe_k)) for pi in order])
        for b in budgets:
            ne = allocate(promise, solved, b, probe_k, kmax, scheme)
            Ki = {pi: probe_k + int(ne[j]) for j, pi in enumerate(order)}
            Kf = int(round(np.mean(list(Ki.values()))))
            for t in taus:
                for pi in order:
                    for a, K in (("fixed", Kf), ("adaptive", Ki[pi])):
                        sel = pools[pi][:K]
                        for pr in (False, True):
                            c, s = sim(sel, pr, warmup, interval, keep_min, w, t)
                            acc[(b, t, a, pr)][pi]["c"].append(c)
                            acc[(b, t, a, pr)][pi]["s"].append(float(s))
    cell = {k: {pi: (float(np.mean(v["c"])), float(np.mean(v["s"]))) for pi, v in d.items()}
            for k, d in acc.items()}
    return cell, tier_of, pidxs, diag


def col(cell, key, pidxs, which):
    j = 0 if which == "c" else 1
    return np.array([cell[key][pi][j] for pi in pidxs])


def boot_mean(vals, B=10000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = vals[rng.integers(0, len(vals), (B, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ratio_diff(num1, den1, num2, den2, B=10000, seed=0):
    """Bootstrap CI of (sum num1/sum den1) - (sum num2/sum den2) resampling prompts (paired)."""
    num1, den1, num2, den2 = map(lambda a: np.asarray(a, float), (num1, den1, num2, den2))
    n = len(num1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    f1 = num1[idx].sum(1) / den1[idx].sum(1)
    f2 = num2[idx].sum(1) / den2[idx].sum(1)
    d = f1 - f2
    point = num1.sum() / den1.sum() - num2.sum() / den2.sum()
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="blend_eval --dump-samples parquet.")
    ap.add_argument("--name", default="pool")
    ap.add_argument("--probe-k", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--scheme", default="frontier")
    ap.add_argument("--budgets", type=float, nargs="+", default=[3, 4, 6, 8])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--prune-interval", type=int, default=32)
    ap.add_argument("--keep-min", type=int, default=2)
    ap.add_argument("--smooth-w", type=int, default=4)
    ap.add_argument("--ref-budget", type=float, default=6.0)
    ap.add_argument("--ref-tau", type=float, default=0.5)
    ap.add_argument("--out-law", default=None, help="Per-tier + pool law-points parquet.")
    args = ap.parse_args()

    dump = pd.read_parquet(args.dump)
    nseed = dump["seed"].nunique()
    cell, tier_of, pidxs, diag = reconstruct(
        dump, args.probe_k, args.kmax, args.scheme, args.budgets, args.taus,
        args.warmup, args.prune_interval, args.keep_min, args.smooth_w)
    P = len(pidxs)
    print(f"\n################ RIGOROUS STATS: {args.name} (n={P} prompts, {nseed} seeds) ################")
    print("All CIs are 95% prompt-level paired bootstraps -> scoped 'on these prompts' "
          "(seed-averaged per prompt; the prompt is the inference unit).")

    # ---- FULL GRID (matched budget: blend at B vs fixed+full at the SAME B) ----
    print("\n=== FULL GRID (no cherry-picking): blend(adapt+prune) vs fixed+full at matched B ===")
    print(f"  {'B':>3} {'tau':>4} {'base_orc':>8} {'blend_orc':>9} {'save%':>6} {'dom?':>5}")
    grid = {}
    for b in args.budgets:
        bf_o = col(cell, (b, args.taus[0], "fixed", False), pidxs, "s").mean()
        bf_c = col(cell, (b, args.taus[0], "fixed", False), pidxs, "c").mean()
        for t in args.taus:
            ap_o = col(cell, (b, t, "adaptive", True), pidxs, "s").mean()
            ap_c = col(cell, (b, t, "adaptive", True), pidxs, "c").mean()
            dom = (ap_o >= bf_o - 1e-9) and (ap_c < bf_c)
            grid[(b, t)] = (bf_o, bf_c, ap_o, ap_c, dom)
            print(f"  {b:>3.0f} {t:>4.1f} {bf_o:>8.3f} {ap_o:>9.3f} {(1 - ap_c / bf_c) * 100:>5.0f}% "
                  f"{'YES' if dom else '-':>5}")

    # ---- HELD-OUT operating-point selection (2-fold) -> unbiased reported number ----
    print("\n=== HELD-OUT operating point (select B,tau on one half, REPORT on the other) ===")
    half = P // 2
    folds = [(pidxs[:half], pidxs[half:]), (pidxs[half:], pidxs[:half])]
    rep_saves, rep_deltas = [], []
    for fi, (sel_ids, rep_ids) in enumerate(folds):
        # select the dominating (b,t) with max saving on the selection half
        best, bestsave = None, -1e9
        for (b, t) in grid:
            so = col(cell, (b, t, "adaptive", True), sel_ids, "s").mean()
            sc = col(cell, (b, t, "adaptive", True), sel_ids, "c").mean()
            bo = col(cell, (b, args.taus[0], "fixed", False), sel_ids, "s").mean()
            bc = col(cell, (b, args.taus[0], "fixed", False), sel_ids, "c").mean()
            if so >= bo - 1e-9 and sc < bc and (1 - sc / bc) > bestsave:
                best, bestsave = (b, t), (1 - sc / bc)
        if best is None:
            print(f"  fold{fi}: no dominating cell on selection half")
            continue
        b, t = best
        # report on held-out half
        rep_apc = col(cell, (b, t, "adaptive", True), rep_ids, "c")
        rep_bfc = col(cell, (b, args.taus[0], "fixed", False), rep_ids, "c")
        rep_apo = col(cell, (b, t, "adaptive", True), rep_ids, "s")
        rep_bfo = col(cell, (b, args.taus[0], "fixed", False), rep_ids, "s")
        sv, slo, shi = boot_ratio_diff(rep_bfc - rep_apc, rep_bfc, np.zeros_like(rep_bfc), np.ones_like(rep_bfc), seed=fi)
        do, dlo, dhi = boot_mean(rep_apo - rep_bfo, seed=fi)
        rep_saves.append(sv)
        rep_deltas.append(do)
        print(f"  fold{fi}: selected B={b:.0f} tau={t:.1f} -> HELD-OUT save {sv * 100:+.0f}% "
              f"[{slo * 100:+.0f},{shi * 100:+.0f}]  oracle-delta {do:+.3f} [{dlo:+.3f},{dhi:+.3f}]")
    if rep_saves:
        print(f"  >>> held-out blend: mean save {np.mean(rep_saves) * 100:+.0f}%  "
              f"mean oracle-delta {np.mean(rep_deltas):+.3f}  (Pareto if delta CI >= 0 and save>0)")

    # ---- reference operating point (B,tau) for the falsifiable mechanism tests ----
    b, t = args.ref_budget, args.ref_tau
    ff_c = col(cell, (b, t, "fixed", False), pidxs, "c")
    ff_s = col(cell, (b, t, "fixed", False), pidxs, "s")
    fp_c = col(cell, (b, t, "fixed", True), pidxs, "c")
    fp_s = col(cell, (b, t, "fixed", True), pidxs, "s")
    af_c = col(cell, (b, t, "adaptive", False), pidxs, "c")
    af_s = col(cell, (b, t, "adaptive", False), pidxs, "s")
    ap_c = col(cell, (b, t, "adaptive", True), pidxs, "c")
    ap_s = col(cell, (b, t, "adaptive", True), pidxs, "s")
    print(f"\n=== FALSIFIABLE MECHANISM TESTS @ ref B={b:.0f} tau={t:.1f} (prompt bootstrap) ===")

    # COMPOUNDING: blend saving vs product-of-levers prediction
    meas = 1 - ap_c.sum() / ff_c.sum()
    pred = 1 - (af_c.sum() / ff_c.sum()) * (fp_c.sum() / ff_c.sum())
    # bootstrap of (measured - product-predicted) saving, resampling prompts
    rng = np.random.default_rng(7)
    idx = rng.integers(0, P, (10000, P))
    m_bs = 1 - ap_c[idx].sum(1) / ff_c[idx].sum(1)
    p_bs = 1 - (af_c[idx].sum(1) / ff_c[idx].sum(1)) * (fp_c[idx].sum(1) / ff_c[idx].sum(1))
    gap = m_bs - p_bs
    print(f"  COMPOUNDING: measured save {meas * 100:.0f}% vs product-predicted {pred * 100:.0f}%  | "
          f"gap {(meas - pred) * 100:+.1f}pp [{np.percentile(gap, 2.5) * 100:+.1f},{np.percentile(gap, 97.5) * 100:+.1f}]"
          f"  ({'COMPOUNDS (gap CI ~0)' if np.percentile(gap, 2.5) * 100 < 0 < np.percentile(gap, 97.5) * 100 else 'gap != 0'})")

    # ALLOCATION-AGNOSTIC: prune token-fraction under fixed vs adaptive
    fp_frac = fp_c.sum() / ff_c.sum()
    ap_frac = ap_c.sum() / af_c.sum()
    pt, plo, phi = boot_ratio_diff(fp_c, ff_c, ap_c, af_c, seed=2)
    print(f"  ALLOCATION-AGNOSTIC: prune keeps {fp_frac * 100:.0f}% cost under fixed vs {ap_frac * 100:.0f}% "
          f"under adaptive | diff {pt * 100:+.1f}pp [{plo * 100:+.1f},{phi * 100:+.1f}]"
          f"  ({'agnostic (CI~0)' if plo < 0 < phi else 'differs'})")

    # SYNERGY: DiD of oracle (does adaptive's lift grow under prune?)
    did = (ap_s - fp_s) - (af_s - ff_s)
    dv, dlo, dhi = boot_mean(did, seed=3)
    print(f"  SYNERGY (DiD oracle): (adapt-fixed)|prune - (adapt-fixed)|full = {dv:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
          f"  ({'lift GROWS under prune' if dlo > 0 else 'not significant' if dlo <= 0 <= dhi else 'shrinks'})")

    # prune accuracy hit (the signed law y-axis, pool-level)
    hit = fp_s - ff_s
    hv, hlo, hhi = boot_mean(hit, seed=4)
    reached = [d for d in diag if d["reached"]]
    pool_auc = _auc([d["correct"] for d in reached], [d["mid"] for d in reached]) if reached else float("nan")
    base_o = float(ff_s.mean())
    print(f"  PRUNE HIT @tau={t:.1f}: oracle {hv:+.3f} [{hlo:+.3f},{hhi:+.3f}] | pool mid-AUC {pool_auc:.3f} "
          f"| base-oracle {base_o:.3f}")

    # ---- per-tier law points (SIGNED hit, all seeds) ----
    law = [{"name": args.name, "level": "pool", "tier": -1, "n": P, "mid_auc": pool_auc,
            "prune_hit": hv, "base_oracle": base_o}]
    tiers = sorted(set(tier_of.values()))
    if len(tiers) > 1:
        print("\n=== PER-TIER law points (signed prune hit; mid-AUC over all seeds) ===")
        for tr in tiers:
            tp = [pi for pi in pidxs if tier_of[pi] == tr]
            if len(tp) < 8:
                continue
            dh = (col(cell, (b, t, "fixed", True), tp, "s") - col(cell, (b, t, "fixed", False), tp, "s"))
            dr = [d for d in diag if d["tier"] == tr and d["reached"]]
            tauc = _auc([d["correct"] for d in dr], [d["mid"] for d in dr]) if dr else float("nan")
            bo = float(col(cell, (b, t, "fixed", False), tp, "s").mean())
            print(f"  tier {tr} (n={len(tp)}): mid-AUC {tauc:.3f}  prune-hit {dh.mean():+.3f}  base-orc {bo:.3f}")
            law.append({"name": args.name, "level": "tier", "tier": tr, "n": len(tp),
                        "mid_auc": tauc, "prune_hit": float(dh.mean()), "base_oracle": bo})
    if args.out_law:
        pd.DataFrame(law).to_parquet(args.out_law, index=False)
        print(f"\n[stats] wrote law points -> {args.out_law}")


if __name__ == "__main__":
    main()
