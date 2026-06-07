"""Shared, torch-free core for the blend study: the faithful offline prune replay (`sim`),
the tie-corrected AUC, and the smoothing — imported by BOTH `blend_eval.py` (measurement) and
`blend_stats.py` (rigorous offline statistics) so the two never diverge.

`sim` replays the EXACT decode-time prune rule from `adaptive_decode.decode_prompt` on a
pre-recorded value trajectory; faithfulness verified by fuzzing (220k cells, 0 mismatches) and
unit tests. Because token-streams are independent under `policy="none"`, truncating a sample at
its prune-point reproduces a live prune run's cost and outcome exactly.
"""
from __future__ import annotations

import numpy as np


def _smooth(hist, w):
    """Trailing mean of the last `w` values (matches adaptive_decode._smooth exactly)."""
    if not hist:
        return 0.0
    tail = hist[-w:]
    return sum(tail) / len(tail)


def _auc(labels, scores):
    """ROC-AUC of `scores` as a ranker of binary `labels` (Mann-Whitney, **tie-corrected** with
    average ranks so all-equal scores give exactly 0.5). NaN scores are dropped."""
    lab = np.asarray(labels, float)
    sc = np.asarray(scores, float)
    m = ~np.isnan(sc)
    lab, sc = lab[m], sc[m]
    npos, nneg = lab.sum(), (1 - lab).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(sc, kind="mergesort")
    sc_s = sc[order]
    ranks_s = np.empty(len(sc), float)
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
            elig = [i for i in act if step < T[i] - 1]     # exclude a sample on its done step
            if len(elig) > keep_min:
                sv = {i: _smooth(vh[i][: step + 1], w) for i in elig}
                for i in sorted(elig, key=lambda j: sv[j])[: len(elig) - keep_min]:
                    if sv[i] < thr:                        # only abandon predicted-losers
                        pruned[i] = True
    solved = any((not pruned[i]) and sel[i]["correct"] for i in range(K))
    return int(cost), bool(solved)
