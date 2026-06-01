"""Token alignment for ZIP-RC targets.

The off-by-one here is SILENT and corrupting if wrong (the head would train on
misaligned targets), so it is unit-tested. Mirrors upstream generate +
train/score exactly:

  gen:   label_positions = range(len(prompt_ids), len(input_ids))   # response tokens
  train: ids = input_ids[:-1][:max_length]
         pos = [p-1 for p in label_positions if 0 <= p-1 < len(ids)]
         tokens_to_completion[k] = len(ids) - pos[k] - 1            # 0 at last prefix
"""
from __future__ import annotations
from typing import List, Sequence, Tuple


def make_label_positions(prompt_token_ids: Sequence[int],
                         output_token_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    input_ids = list(prompt_token_ids) + list(output_token_ids)
    label_positions = list(range(len(prompt_token_ids), len(input_ids)))
    return input_ids, label_positions


def train_time_align(input_ids: Sequence[int], label_positions: Sequence[int],
                     max_length: int = 2048):
    """Returns (ids, positions, tokens_to_completion)."""
    ids = list(input_ids)[:-1][:max_length]
    n = len(ids)
    pos = [p - 1 for p in label_positions if 0 <= p - 1 < n]
    ttc = [n - p - 1 for p in pos]
    return ids, pos, ttc
