"""Online probe-and-reallocate, stage 3: evaluate the end-to-end result against a fixed-K
baseline at MATCHED mean budget.

  adaptive: probe (first --probe-k of the existing set) + FRESH extras (gen'd via n_extra).
            oracle_i = probe-solved OR any-extra-correct ; cost_i = probe_k + n_extra_i
  fixed   : pass@K from the same existing set, K = round(mean adaptive cost).
This isolates whether reallocating budget by the mid-trajectory signal beats spending it
uniformly, with REAL fresh generation and true compute accounting.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-set", required=True, help="Labeled set (>= kmax samples/prompt); first probe-k are the probe, also the fixed-K pool.")
    ap.add_argument("--extra-set", required=True, help="Labeled FRESH extra rollouts (from gen --n-col).")
    ap.add_argument("--probe-k", type=int, default=2)
    args = ap.parse_args()

    base = pd.read_parquet(args.probe_set)
    extra = pd.read_parquet(args.extra_set)
    extra_by = {int(pi): g["correct"].astype(float).values for pi, g in extra.groupby("prompt_idx")}

    orc_adapt, cost_adapt, base_pool = [], [], {}
    for pi, g in base.groupby("prompt_idx"):
        c = g["correct"].astype(float).values
        base_pool[int(pi)] = c
        probe_solved = bool(c[: args.probe_k].max() >= 1.0)
        ex = extra_by.get(int(pi), np.array([]))
        orc_adapt.append(float(probe_solved or (ex.max() >= 1.0 if ex.size else False)))
        cost_adapt.append(args.probe_k + int(ex.size))

    mean_cost = float(np.mean(cost_adapt))
    k_fixed = int(round(mean_cost))
    orc_fixed = [float(c[:k_fixed].max() >= 1.0) for c in base_pool.values()]

    print(f"=== online probe-and-reallocate ({len(orc_adapt)} prompts) ===")
    print(f"  adaptive : oracle {np.mean(orc_adapt):.3f}  | mean cost {mean_cost:.2f} samples/prompt")
    print(f"  fixed@{k_fixed}  : oracle {np.mean(orc_fixed):.3f}  | cost {k_fixed} samples/prompt (matched)")
    print(f"  >>> adaptive − fixed @ matched budget: {np.mean(orc_adapt) - np.mean(orc_fixed):+.3f} oracle")


if __name__ == "__main__":
    main()
