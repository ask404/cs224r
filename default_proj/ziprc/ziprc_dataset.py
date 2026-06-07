"""ZIP-RC training dataset (joint reward/length bin labels).

Adapted from github.com/rohinmanvi/ZIP-RC (src/ziprc_dataset.py) with three
Countdown/0.5B changes:
  1. length_bins is a constructor arg (default = short Countdown bins), instead of
     the hardcoded 32k-token bins (which would be degenerate here).
  2. label_column may be ANY column (e.g. 'correct', 'reward3', 'value'); the
     reward is read as a float in [0,1] (binary 'correct' still validated to {0,1}).
  3. Binning delegates to ziprc.grid (the unit-tested implementation) so dataset,
     scorer, and tests share one source of truth.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ziprc.config import LENGTH_BINS_COUNTDOWN, REWARD_VALUES_STRUCTURED  # noqa: E402
from ziprc.grid import bin_index, value_bin_edges  # noqa: E402


def _to_int_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(t) for t in x]
    if isinstance(x, str):
        return [int(t) for t in ast.literal_eval(x)]
    return [int(t) for t in list(x)]


class ZIPDataset(Dataset):
    def __init__(
        self,
        table,
        max_length: int = 2048,
        reward_values: Optional[Sequence[float]] = None,
        length_bins: Optional[Sequence[int]] = None,
        label_column: str = "reward3",
    ):
        self.table = pq.read_table(table).to_pandas() if isinstance(table, str) else table
        cols = set(self.table.columns)
        if label_column not in cols:
            raise ValueError(f"label_column '{label_column}' not in data columns: {sorted(cols)}")
        self.reward_column = label_column

        if label_column == "correct":
            uniq = {float(x) for x in self.table["correct"].dropna().unique().tolist()}
            if not uniq.issubset({0.0, 1.0}):
                raise ValueError(f"'correct' must be binary; found {sorted(uniq)}")
        self.table[self.reward_column] = self.table[self.reward_column].astype(float)

        self.max_length = max_length
        self.length_bins = list(length_bins) if length_bins is not None else list(LENGTH_BINS_COUNTDOWN)
        self.num_length_bins = len(self.length_bins) - 1
        self.reward_values = list(reward_values) if reward_values is not None else list(REWARD_VALUES_STRUCTURED)
        self.num_reward_states = len(self.reward_values)
        self.num_bins = self.num_length_bins * self.num_reward_states
        self.value_bin_edges = value_bin_edges(self.reward_values)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        ids_list = _to_int_list(row["input_ids"])
        ids = torch.tensor(ids_list, dtype=torch.long)[:-1][: self.max_length]

        lp_list = _to_int_list(row["label_positions"])
        label_positions = [p - 1 for p in lp_list if 0 <= p - 1 < len(ids)]

        reward_value = float(row[self.reward_column])
        reward_value = min(1.0, max(0.0, reward_value))

        total_length = len(ids)
        bin_labels = []
        for pos in label_positions:
            tokens_to_completion = total_length - pos - 1
            bin_labels.append(
                bin_index(tokens_to_completion, reward_value, self.reward_values, self.length_bins)
            )

        return {
            "input_ids": ids,
            "label_positions": label_positions,
            "bin_labels": bin_labels,
            "num_bins": self.num_bins,
        }

    @staticmethod
    def collate_fn(batch):
        max_len = max(s["input_ids"].size(0) for s in batch)
        return {
            "input_ids": torch.stack(
                [F.pad(s["input_ids"], (0, max_len - s["input_ids"].size(0))) for s in batch]
            ),
            "label_positions": [s["label_positions"] for s in batch],
            "bin_labels": [s["bin_labels"] for s in batch],
            "num_bins": batch[0]["num_bins"],
        }
