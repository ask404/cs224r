"""Joint (reward_state x length_bin) grid math.

Mirrors upstream ZIPDataset bin construction EXACTLY so vendored training/scoring
stay compatible, plus inverse/marginalization helpers used by tests and offline
value computation.

Bin-index convention (upstream): idx = length_bin + reward_state * num_length_bins
so reshape to (num_reward_states, num_length_bins) is correct, and length is the
fastest-varying axis within each reward block.
"""
from __future__ import annotations
from collections import Counter  # noqa: F401  (kept for parity / future use)
from typing import List, Sequence

import numpy as np


def value_bin_edges(reward_values: Sequence[float]) -> List[float]:
    """Midpoint edges between consecutive reward_values, clamped to [0,1].
    Identical to ZIPDataset.value_bin_edges."""
    rv = list(reward_values)
    R = len(rv)
    if R >= 2:
        edges = [0.0] * (R + 1)
        for i in range(1, R):
            edges[i] = 0.5 * (rv[i - 1] + rv[i])
        first_step = rv[1] - rv[0]
        last_step = rv[-1] - rv[-2]
        edges[0] = rv[0] - 0.5 * first_step
        edges[-1] = rv[-1] + 0.5 * last_step
    else:
        edges = [rv[0] - 0.5, rv[0] + 0.5]
    edges[0] = max(0.0, edges[0])
    edges[-1] = min(1.0, edges[-1])
    return edges


def length_bin_of(tokens_to_completion: int, length_bins: Sequence[int]) -> int:
    n = len(length_bins) - 1
    for i in range(n):
        if length_bins[i] <= tokens_to_completion < length_bins[i + 1]:
            return i
    if tokens_to_completion >= length_bins[-1]:
        return n - 1
    return 0  # ttc < length_bins[0]; with bins[0]=0 and ttc>=0 this won't trigger


def reward_state_of(reward: float, edges: Sequence[float]) -> int:
    R = len(edges) - 1
    for i in range(R):
        if edges[i] <= reward < edges[i + 1]:
            return i
    if reward >= edges[-1]:
        return R - 1
    return 0


def bin_index(tokens_to_completion: int, reward: float,
              reward_values: Sequence[float], length_bins: Sequence[int]) -> int:
    L = len(length_bins) - 1
    lb = length_bin_of(tokens_to_completion, length_bins)
    rs = reward_state_of(reward, value_bin_edges(reward_values))
    return lb + rs * L


def decompose(idx: int, num_length_bins: int):
    """idx -> (reward_state, length_bin)."""
    return idx // num_length_bins, idx % num_length_bins


def marginal_reward(probs, num_reward_states: int, num_length_bins: int):
    """probs[..., R*L] -> [..., R], summing over length bins."""
    p = np.asarray(probs, dtype=float)
    p = p.reshape(*p.shape[:-1], num_reward_states, num_length_bins)
    return p.sum(axis=-1)


def marginal_length(probs, num_reward_states: int, num_length_bins: int):
    """probs[..., R*L] -> [..., L], summing over reward states."""
    p = np.asarray(probs, dtype=float)
    p = p.reshape(*p.shape[:-1], num_reward_states, num_length_bins)
    return p.sum(axis=-2)


def expected_value(probs, reward_values: Sequence[float], length_bins: Sequence[int]):
    """E[reward] = sum_r P(reward_state=r) * reward_values[r]."""
    R = len(reward_values)
    L = len(length_bins) - 1
    pr = marginal_reward(probs, R, L)
    return np.asarray(pr) @ np.asarray(reward_values, dtype=float)
