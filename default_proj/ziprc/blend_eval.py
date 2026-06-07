#!/usr/bin/env python3
"""Blend test: does **adaptive-K** (across-prompt allocation) compose with **prune**
(within-sample compute) so the savings *compound*?

The two levers act on different quantities:
  * adaptive-K changes ORACLE at ~matched sample budget (spend whole samples where they help);
  * prune changes COST (tokens) at ~matched oracle (kill predicted-loser streams mid-generation).
"Compounding" means the blend captures BOTH, ~independently: prune saves the same token-fraction
under either allocation, AND adaptive's oracle lift survives pruning (it doesn't kill the very
frontier samples adaptive paid for — failure mode #4).

Method (faithful + cheap, ONE generation per prompt):
  1. Generate a pool of `pool_k` samples with policy="none" (no real pruning), recording each
     sample's value trajectory `vhist`, length, correctness, and the 25%-through value.
  2. Allocate ONLINE from THIS pool's probe (first probe_k samples) via the SAME `allocate()`
     the offline allocator uses — so adaptive is self-consistent (no stale allocation).
  3. Replay the EXACT decode-time prune rule OFFLINE over a threshold sweep to fill the full
     adaptive x prune 2x2 per threshold. Faithful because token-streams are independent
     (pruning sample i never changes j), so truncating i at its prune-point is exact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from evaluation.countdown import compute_score  # noqa: E402
from ziprc.adaptive_decode import AdaptiveDecoder, _smooth  # noqa: E402
from ziprc.allocate_budget import allocate  # noqa: E402
from ziprc.config import (DISTRIBUTION_TOKEN_ID, LENGTH_BINS_COUNTDOWN,  # noqa: E402
                          REWARD_VALUES_BINARY)


def sim(sel, prune, warmup, interval, keep_min, w, thr):
    """Replay decode_prompt's loop on pre-recorded trajectories for a selected sample subset.
    sel: list of dicts {vhist:list, n_tokens:int(=len vhist), correct:bool}.
    Returns (cost = active forward passes, solved = any surviving-correct)."""
    K = len(sel)
    if K == 0:
        return 0, False
    T = [s["n_tokens"] for s in sel]
    vh = [s["vhist"] for s in sel]
    if not prune:
        return int(sum(T)), bool(any(s["correct"] for s in sel))
    pruned = [False] * K
    cost = 0
    for step in range(max(T)):
        act = [i for i in range(K) if step < T[i] and not pruned[i]]
        if not act:
            break
        cost += len(act)                                   # one forward pass per active sample
        if step >= warmup and step % interval == 0:
            # decode-time prune candidates = active AND still generating (not on their done token)
            elig = [i for i in act if step < T[i] - 1]
            if len(elig) > keep_min:
                sv = {i: _smooth(vh[i][: step + 1], w) for i in elig}
                for i in sorted(elig, key=lambda j: sv[j])[: len(elig) - keep_min]:
                    if sv[i] < thr:                        # only abandon predicted-losers
                        pruned[i] = True
    solved = any((not pruned[i]) and sel[i]["correct"] for i in range(K))
    return int(cost), bool(solved)


def _auc(labels, scores):
    """ROC-AUC of `scores` as a ranker of binary `labels` (Mann-Whitney form, **tie-corrected**
    with average ranks so all-equal scores give exactly 0.5). NaN scores are dropped."""
    lab = np.asarray(labels, float)
    sc = np.asarray(scores, float)
    m = ~np.isnan(sc)
    lab, sc = lab[m], sc[m]
    npos, nneg = lab.sum(), (1 - lab).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(sc, kind="mergesort")
    sc_s = sc[order]
    ranks_s = np.empty(len(sc), float)        # 1-based AVERAGE ranks within tie-groups
    i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and sc_s[j + 1] == sc_s[i]:
            j += 1
        ranks_s[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank = np.empty(len(sc), float)
    rank[order] = ranks_s
    return float((rank[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Trained head-only model dir.")
    ap.add_argument("--prompts", required=True, help="Parquet with one row/prompt (prompt,target,nums[,prompt_idx]).")
    ap.add_argument("--probe-k", type=int, default=2)
    ap.add_argument("--pool-k", type=int, default=8, help="Samples generated per prompt (>= kmax).")
    ap.add_argument("--budget", type=float, default=6.0, help="Target mean total samples/prompt for adaptive.")
    ap.add_argument("--budgets", type=float, nargs="+", default=None,
                    help="If set, sweep these budgets offline from one generation -> Pareto frontier.")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--scheme", choices=["frontier", "promise"], default="frontier")
    ap.add_argument("--num-prompts", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--prune-interval", type=int, default=32)
    ap.add_argument("--keep-min", type=int, default=2)
    ap.add_argument("--smooth-w", type=int, default=4)
    ap.add_argument("--prune-thresholds", type=float, nargs="+", default=[0.5, 0.4, 0.3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="If set, repeat the whole online run at each seed and report mean±std "
                         "+ per-seed Pareto-dominance count (firms up small-n pools).")
    ap.add_argument("--out", default=None, help="Threshold-sweep summary parquet.")
    ap.add_argument("--out-tier", default=None, help="Per-tier calibration-gradient parquet.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to(dev).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    dec = AdaptiveDecoder(model, tok, DISTRIBUTION_TOKEN_ID, list(REWARD_VALUES_BINARY),
                          list(LENGTH_BINS_COUNTDOWN), dev)

    df = pd.read_parquet(args.prompts)
    if "prompt_idx" in df.columns:
        df = df.drop_duplicates("prompt_idx")
    df = df.reset_index(drop=True)
    if "target" not in df.columns or "nums" not in df.columns:   # main-pool parquet stores ground_truth
        gt = df["ground_truth"]
        df = df.assign(
            target=[int(g["target"]) for g in gt],
            nums=[[int(x) for x in (g["numbers"] if "numbers" in g else g["nums"])] for g in gt],
        )
    if args.num_prompts and args.num_prompts < len(df):
        df = df.iloc[: args.num_prompts].reset_index(drop=True)

    budgets = args.budgets if args.budgets else [args.budget]

    def gen_pools(seed):
        """Generate the pool_k-sample pool ONCE per prompt for this seed (the only GPU cost).
        Budgets and thresholds are then swept offline by re-selecting/re-pruning these pools."""
        pools, promise, solved, diag, tiers = [], [], [], [], []
        for idx, r in enumerate(df.itertuples(index=False)):
            gt = {"target": int(r.target), "numbers": [int(x) for x in r.nums]}
            tier = len(list(r.nums))                          # operand count = difficulty tier
            torch.manual_seed(seed * 100003 + int(getattr(r, "prompt_idx", idx)))
            o = dec.decode_prompt(r.prompt, args.pool_k, args.max_new_tokens, args.temperature,
                                  args.top_p, args.top_k, policy="none",
                                  warmup=args.warmup, prune_interval=args.prune_interval,
                                  keep_min=args.keep_min, smooth_w=args.smooth_w)
            pool = []
            for s in o["samples"]:
                vh = s["vhist"]
                q25 = vh[len(vh) // 4] if vh else 0.0        # value 25% through the trajectory
                corr = compute_score(s["text"], gt) == 1.0
                pool.append({"vhist": vh, "n_tokens": s["n_tokens"], "correct": corr, "q25": q25})
                reached = s["n_tokens"] > args.warmup
                diag.append({"reached": reached, "correct": corr, "tier": tier,
                             "mid": _smooth(vh[: args.warmup + 1], args.smooth_w) if reached else float("nan")})
            pools.append(pool)
            tiers.append(tier)
            probe = pool[: args.probe_k]
            promise.append(float(np.mean([p["q25"] for p in probe])) if probe else 0.0)
            solved.append(bool(any(p["correct"] for p in probe)))
        return pools, promise, np.asarray(solved), diag, tiers

    def eval_budget(pools, promise, solved, budget):
        """Offline: allocate at `budget`, then fill the (tau x arm x prune) grid."""
        n_extra = allocate(promise, solved, budget, args.probe_k, args.kmax, args.scheme)
        K_i = args.probe_k + n_extra.astype(int)
        K_f = int(round(K_i.mean()))
        out = {"K_f": K_f, "meanK": float(K_i.mean())}
        for thr in args.prune_thresholds:
            cells = {(a, p): {"cost": [], "orc": []} for a in ("fixed", "adaptive") for p in (False, True)}
            for j, pool in enumerate(pools):
                for arm, K in (("fixed", K_f), ("adaptive", int(K_i[j]))):
                    selp = pool[:K]
                    for prune in (False, True):
                        c, s = sim(selp, prune, args.warmup, args.prune_interval, args.keep_min, args.smooth_w, thr)
                        cells[(arm, prune)]["cost"].append(c)
                        cells[(arm, prune)]["orc"].append(float(s))
            for a in ("fixed", "adaptive"):
                for p in (False, True):
                    out[(thr, a, p, "orc")] = float(np.mean(cells[(a, p)]["orc"]))
                    out[(thr, a, p, "cost")] = float(np.mean(cells[(a, p)]["cost"]))
        return out

    seeds = args.seeds if args.seeds else [args.seed]
    runs = []
    stash = None
    for si, sd in enumerate(seeds):
        pools, promise, solved, diag, tiers = gen_pools(sd)
        if stash is None:
            stash = (pools, promise, solved, tiers)           # seed-0 pools for per-tier 2x2
        rb = {"diag": diag, "solved": int(solved.sum())}
        for b in budgets:
            rb[b] = eval_budget(pools, promise, solved, b)
        runs.append(rb)
        print(f"[blend] seed {sd} done ({si + 1}/{len(seeds)}) | budgets={budgets} "
              f"probe-solved={rb['solved']}/{len(df)}", flush=True)

    # --- CALIBRATION DIAGNOSTIC: why prune's accuracy-cost is (un)recoverable on this pool ---
    # Among samples that reach the prune window, can the smoothed mid-value separate eventual
    # winners from losers? And where do the at-risk WINNERS sit relative to the threshold?
    alld = [d for r in runs for d in r["diag"]]
    reached = [d for d in alld if d["reached"]]
    mid_auc = _auc([d["correct"] for d in reached], [d["mid"] for d in reached]) if reached else float("nan")
    win_mid = np.array([d["mid"] for d in reached if d["correct"]])
    los_mid = np.array([d["mid"] for d in reached if not d["correct"]])
    print(f"\n=== CALIBRATION DIAGNOSTIC (n={len(alld)} samples, {len(reached)} reach prune window @ step {args.warmup}) ===")
    print(f"  mid-value AUC (winner vs loser separability at prune point): {mid_auc:.3f}")
    if win_mid.size and los_mid.size:
        print(f"  eventual WINNERS' mid-value: median {np.median(win_mid):.3f}  | "
              f">=0.5 (prune-safe) {np.mean(win_mid >= 0.5) * 100:.0f}%  "
              f"[0.3,0.5) (tau-recoverable) {np.mean((win_mid >= 0.3) & (win_mid < 0.5)) * 100:.0f}%  "
              f"<0.3 (unrecoverable) {np.mean(win_mid < 0.3) * 100:.0f}%")
        print(f"  eventual LOSERS'  mid-value: median {np.median(los_mid):.3f}  | "
              f"<0.5 (correctly killable) {np.mean(los_mid < 0.5) * 100:.0f}%")
        recoverable = float(np.mean((win_mid >= 0.3) & (win_mid < 0.5)))
        print(f"  >>> of winners at risk at tau=0.5, {recoverable * 100:.0f}% are RECOVERABLE by tau=0.4 "
              f"(borderline 0.3-0.5) -> predicts whether the blend reaches Pareto-dominance")

    # --- PER-TIER calibration law: each operand-count tier is a different calibration level, so
    # one pool yields a CALIBRATION GRADIENT. For each tier: mid-AUC (separability) vs the blend's
    # best Pareto compute-saving at >= that tier's baseline oracle. Predicts the free lunch tier-wise.
    tier_rows = []
    present = sorted(set(d["tier"] for d in alld))
    if len(present) > 1:
        pools0, promise0, solved0, tiers0 = stash
        prim = args.budget
        ne0 = allocate(promise0, solved0, prim, args.probe_k, args.kmax, args.scheme)
        Ki0 = args.probe_k + ne0.astype(int)
        Kf0 = int(round(Ki0.mean()))
        print(f"\n=== PER-TIER CALIBRATION GRADIENT (budget={prim:g}, seed-0 pools) ===")
        print(f"  {'tier':<6} {'n':>4} {'mid-AUC':>8} {'base-orc':>9} {'blend Pareto-save':>18}")
        for t in present:
            dt = [d for d in alld if d["tier"] == t and d["reached"]]
            tauc = _auc([d["correct"] for d in dt], [d["mid"] for d in dt]) if dt else float("nan")
            idxs = [j for j in range(len(tiers0)) if tiers0[j] == t]
            ff_o = float(np.mean([any(s["correct"] for s in pools0[j][:Kf0]) for j in idxs]))
            ff_c = float(np.mean([sum(s["n_tokens"] for s in pools0[j][:Kf0]) for j in idxs]))
            best = 0.0
            for thr in args.prune_thresholds:
                ap = [sim(pools0[j][: int(Ki0[j])], True, args.warmup, args.prune_interval,
                          args.keep_min, args.smooth_w, thr) for j in idxs]
                ap_o = float(np.mean([s for _, s in ap]))
                ap_c = float(np.mean([c for c, _ in ap]))
                if ap_o >= ff_o - 1e-9 and ap_c < ff_c:
                    best = max(best, (1 - ap_c / ff_c) * 100)
            print(f"  {t:<6} {len(idxs):>4} {tauc:8.3f} {ff_o:9.3f} {best:+17.0f}%")
            tier_rows.append({"tier": t, "n": len(idxs), "mid_auc": tauc,
                              "base_oracle": ff_o, "pareto_saving_pct": best})

    def agg(b, thr, a, p, field):
        v = [r[b][(thr, a, p, field)] for r in runs]
        return float(np.mean(v)), float(np.std(v))

    print(f"\n=== PARETO FRONTIER: fixed+full vs blend ({len(seeds)} seeds, n={len(df)} prompts, "
          f"budgets={budgets}, tau={args.prune_thresholds}) ===")
    summary = []
    best = None  # best Pareto-dominating blend point across the whole grid
    for b in budgets:
        # fixed+full is prune-independent -> read from the first threshold
        bf_o, bf_os = agg(b, args.prune_thresholds[0], "fixed", False, "orc")
        bf_c, _ = agg(b, args.prune_thresholds[0], "fixed", False, "cost")
        print(f"\n-- budget B={b} | fixed+full: oracle {bf_o:.3f}±{bf_os:.3f}  cost {bf_c:.0f} --")
        for thr in args.prune_thresholds:
            ap_o, ap_os = agg(b, thr, "adaptive", True, "orc")
            ap_c, _ = agg(b, thr, "adaptive", True, "cost")
            fp_o, _ = agg(b, thr, "fixed", True, "orc")
            fp_c, _ = agg(b, thr, "fixed", True, "cost")
            af_o, _ = agg(b, thr, "adaptive", False, "orc")
            af_c, _ = agg(b, thr, "adaptive", False, "cost")
            dom = [(r[b][(thr, "adaptive", True, "orc")] >= r[b][(thr, "fixed", False, "orc")] - 1e-9)
                   and (r[b][(thr, "adaptive", True, "cost")] < r[b][(thr, "fixed", False, "cost")]) for r in runs]
            ndom = int(sum(dom))
            star = " *PARETO*" if (ap_o >= bf_o - 1e-9 and ap_c < bf_c) else ""
            print(f"   tau={thr:g}: blend(adapt+prune) oracle {ap_o:.3f}±{ap_os:.3f} "
                  f"cost {ap_c:.0f} ({(1 - ap_c / bf_c) * 100:+.0f}%)  dom {ndom}/{len(seeds)}{star}")
            if star and (best is None or ap_o - best["adapt_prune_orc"] > 1e-9
                         or (abs(ap_o - best["adapt_prune_orc"]) <= 1e-9 and ap_c < best["adapt_prune_cost"])):
                best = {"B": b, "tau": thr, "adapt_prune_orc": ap_o, "adapt_prune_cost": ap_c,
                        "fixed_full_orc": bf_o, "fixed_full_cost": bf_c, "pareto_seeds": ndom}
            summary.append({"budget": b, "tau": thr, "n_seeds": len(seeds), "n_prompts": len(df),
                            "mid_auc": mid_auc,
                            "fixed_full_orc": bf_o, "fixed_full_cost": bf_c,
                            "fixed_prune_orc": fp_o, "fixed_prune_cost": fp_c,
                            "adapt_full_orc": af_o, "adapt_full_cost": af_c,
                            "adapt_prune_orc": ap_o, "adapt_prune_orc_std": ap_os, "adapt_prune_cost": ap_c,
                            "pareto_seeds": ndom})
    if best:
        print(f"\n  >>> BEST Pareto-dominating blend: B={best['B']} tau={best['tau']} -> "
              f"oracle {best['adapt_prune_orc']:.3f} (>= fixed+full {best['fixed_full_orc']:.3f}) at "
              f"{(1 - best['adapt_prune_cost'] / best['fixed_full_cost']) * 100:+.0f}% compute, "
              f"{best['pareto_seeds']}/{len(seeds)} seeds")
    else:
        print("\n  >>> no blend point Pareto-dominates fixed+full on this pool (compute-for-accuracy trade only)")

    if args.out:
        pd.DataFrame(summary).to_parquet(args.out, index=False)
        print(f"\n[blend] wrote frontier grid -> {args.out}")
    if args.out_tier and tier_rows:
        pd.DataFrame(tier_rows).to_parquet(args.out_tier, index=False)
        print(f"[blend] wrote per-tier gradient -> {args.out_tier}")


if __name__ == "__main__":
    main()
