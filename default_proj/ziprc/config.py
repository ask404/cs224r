"""Central config for the ZIP-RC-Lite (Countdown / Qwen2.5-0.5B) extension.

Deliberately small/cheap defaults for a 0.5B model on the short-horizon
Countdown task. See ziprc/README.md for rationale.
"""
from __future__ import annotations
from typing import Sequence

# --- Reserved-logit slice (Qwen2.5-0.5B) ----------------------------------
# config.vocab_size = 151936, but the tokenizer only NAMES ids up to 151664,
# leaving ids 151665..151935 (271 unused embedding rows). We park the ZIP-RC
# joint-distribution head in the first contiguous slice of these.
#
# NOTE: the upstream repo default 151669 is a *Qwen3* id; do NOT use it here.
VOCAB_SIZE = 151936
HIGHEST_REAL_TOKEN_ID = 151664
FIRST_RESERVED_ID = 151665
DISTRIBUTION_TOKEN_ID = FIRST_RESERVED_ID

# --- Length bins (REMAINING tokens to completion) -------------------------
# Upstream defaults ([0,256,...,32768]) target 32k-token math traces and are a
# TRAP for Countdown (responses < ~300 tokens => ~all mass in bin 0, dead length
# signal). These short bins keep resolution where it matters. RE-AUDIT on real
# rollouts (M3, `ziprc.audit`) and adjust before scaling.
LENGTH_BINS_COUNTDOWN = [0, 16, 32, 64, 128, 256, 512, 1024]  # 7 bins

# --- Reward states --------------------------------------------------------
# Binary baseline vs structured 3-outcome verifier. Same rollouts, two labels.
REWARD_VALUES_BINARY = [0.0, 1.0]
REWARD_VALUES_STRUCTURED = [0.0, 0.1, 1.0]  # incoherent / coherent / correct

# Semantic outcome -> reward value (structured verifier)
INCOHERENT = 0.0
COHERENT = 0.1
CORRECT = 1.0


def num_length_bins(length_bins: Sequence[int] = LENGTH_BINS_COUNTDOWN) -> int:
    return len(length_bins) - 1


def num_bins(reward_values: Sequence[float],
             length_bins: Sequence[int] = LENGTH_BINS_COUNTDOWN) -> int:
    return len(reward_values) * num_length_bins(length_bins)


def assert_slice_fits(reward_values: Sequence[float],
                      length_bins: Sequence[int] = LENGTH_BINS_COUNTDOWN,
                      start: int = DISTRIBUTION_TOKEN_ID,
                      vocab_size: int = VOCAB_SIZE) -> int:
    """Validate the reserved slice [start, start+num_bins) is in-bounds and does
    not collide with real tokens. Returns num_bins."""
    nb = num_bins(reward_values, length_bins)
    end = start + nb
    if start <= HIGHEST_REAL_TOKEN_ID:
        raise ValueError(
            f"distribution_token_id {start} collides with real tokens "
            f"(<= {HIGHEST_REAL_TOKEN_ID})."
        )
    if end > vocab_size:
        raise ValueError(
            f"Reserved slice [{start},{end}) exceeds vocab {vocab_size}; "
            f"reduce bins or lower distribution_token_id."
        )
    return nb
