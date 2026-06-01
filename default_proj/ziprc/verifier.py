"""Three-outcome structured verifier for Countdown.

  outcome     reward  how decided
  ----------  ------  -----------------------------------------------------------
  correct      1.0    rule-based compute_score == 1.0            (deterministic)
  coherent     0.1    rule-based FAIL, but the trace shows a valid arithmetic step
                      over the allowed numbers                   (judge/heuristic)
  incoherent   0.0    rule-based FAIL and no valid arithmetic toward the target

Bins 1 & 3 are deterministic; only the failure set is split by a judge, which
bounds judge cost and keeps the high-stakes correctness call deterministic.

Two coherence backends:
  - coherence_heuristic(...): cheap, deterministic, NO API. Unblocks the whole
    pipeline (binary viability needs no judge at all) and is the unit-test target.
  - HaikuCoherenceJudge: the real LLM judge (Claude Haiku), cached, failures-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.countdown import compute_score, evaluate_equation  # noqa: E402
from ziprc.config import COHERENT, CORRECT, INCOHERENT  # noqa: E402

_NUM = re.compile(r"\d+")
# Candidate arithmetic spans: start/end on a digit, only arithmetic chars between.
_EXPR = re.compile(r"[0-9][0-9+\-*/().\s]*[0-9]")


def _multiset_subset(used, available) -> bool:
    cu, ca = Counter(used), Counter(available)
    return all(cu[k] <= ca[k] for k in cu)


def coherence_heuristic(response: str, ground_truth: dict) -> bool:
    """True => 'coherent'. Coherent if the trace contains >=1 arithmetic
    expression that (a) uses only a subset (with multiplicity) of the allowed
    numbers and (b) evaluates to a finite number. Incoherent = degenerate text,
    wrong numbers, or no real arithmetic."""
    numbers = [int(n) for n in ground_truth["numbers"]]
    for span in _EXPR.findall(response):
        span = span.strip()
        used = [int(n) for n in _NUM.findall(span)]
        if len(used) < 2:
            continue  # need an actual operation between >=2 numbers
        if not _multiset_subset(used, numbers):
            continue
        if evaluate_equation(span) is not None:
            return True
    return False


def three_outcome_label(response: str, ground_truth: dict, judge=None) -> float:
    """Return one of {INCOHERENT=0.0, COHERENT=0.1, CORRECT=1.0}."""
    if compute_score(response, ground_truth) == CORRECT:
        return CORRECT
    if judge is None:
        coherent = coherence_heuristic(response, ground_truth)
    else:
        coherent = judge.classify(response, ground_truth)
    return COHERENT if coherent else INCOHERENT


def binary_label(response: str, ground_truth: dict) -> float:
    """Collapsed binary reward (format/incoherent both -> 0.0)."""
    return 1.0 if compute_score(response, ground_truth) == CORRECT else 0.0


# --------------------------------------------------------------------------- #
# Claude Haiku coherence judge (failures-only, cached, prompt-cached)
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM = """You grade the REASONING COHERENCE of a failed attempt at a \
Countdown arithmetic puzzle. In Countdown, the solver must combine ALL the given \
numbers exactly once with +, -, *, / to hit a target. This attempt did NOT reach \
the target (it already failed the rule-based check). Your ONLY job is to decide \
whether the reasoning is COHERENT or INCOHERENT.

COHERENT: the trace performs genuine arithmetic using the given numbers and makes \
at least one valid step on a plausible path toward the target (even if the final \
answer is wrong, or it gives up partway). A "coherent failure" looks like real \
problem-solving that didn't land.

INCOHERENT: degenerate or off-task text, hallucinated/wrong numbers, no real \
arithmetic, repetition/gibberish, or arithmetic that is internally nonsensical.

Answer with EXACTLY one word: COHERENT or INCOHERENT."""


class HaikuCoherenceJudge:
    """Calls Claude Haiku to split failures into coherent vs incoherent.

    - Only invoke on rule-based FAILURES (caller's responsibility).
    - Disk-cached by sha1(ground_truth + response) so re-runs are free.
    - Uses prompt caching on the static rubric (system block) to cut input cost.
    """

    def __init__(self, model: Optional[str] = None, cache_path: Optional[str] = None,
                 max_tokens: int = 8):
        import anthropic  # lazy: only needed when actually judging
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ZIPRC_JUDGE_MODEL", "claude-haiku-4-5")
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self._cache = json.load(open(cache_path))
            except Exception:
                self._cache = {}
        self.n_api_calls = 0

    @staticmethod
    def _key(response: str, ground_truth: dict) -> str:
        blob = json.dumps({"gt": ground_truth, "r": response}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()

    def _persist(self):
        if self.cache_path:
            tmp = f"{self.cache_path}.tmp"
            json.dump(self._cache, open(tmp, "w"))
            os.replace(tmp, self.cache_path)

    def classify(self, response: str, ground_truth: dict) -> bool:
        """True => coherent."""
        key = self._key(response, ground_truth)
        if key in self._cache:
            return self._cache[key]

        user = (
            f"Target: {ground_truth['target']}\n"
            f"Allowed numbers: {list(ground_truth['numbers'])}\n\n"
            f"Attempt:\n{response}\n\n"
            f"COHERENT or INCOHERENT?"
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{
                "type": "text",
                "text": _JUDGE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
        )
        self.n_api_calls += 1
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").upper()
        coherent = "COHERENT" in text and "INCOHERENT" not in text
        self._cache[key] = coherent
        if self.n_api_calls % 50 == 0:
            self._persist()
        return coherent

    def flush(self):
        self._persist()
