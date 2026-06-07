#!/usr/bin/env python3
"""Generate a WIDE-DIFFICULTY Countdown dataset to test the difficulty-variance
hypothesis for adaptive-K (see ADAPTIVE_K.md / EXPERIMENT_difficulty.md).

Difficulty is controlled by OPERAND CARDINALITY (n_numbers in {3,4,5,6}): more numbers
=> larger search space => harder. Each problem is generated FROM a random valid solution
tree, so it is solvable by construction; we self-verify with the project's own
`compute_score`. Output schema matches the training pipeline: prompt, target, nums,
ground_truth, plus `n_numbers` as a clean difficulty label for stratified analysis.

Pure-Python generation (testable locally); pandas/datasets only needed to write/push.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from evaluation.countdown import compute_score  # noqa: E402

_PROMPT_TMPL = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in the "
    "mind and then provides the user with the answer.\nUser: Using the numbers [{nums}], "
    "create an equation that equals {target}. You can use basic arithmetic operations "
    "(+, -, *, /) and each number can only be used once. Show your work in <think> "
    "</think> tags. And return the final answer in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step."
    "<|im_end|>\n<|im_start|>assistant\n"
)


def _combine(a, b, op):
    """Return (value, expr) for a `op` b, or None if invalid (non-integer/negative)."""
    av, ae = a
    bv, be = b
    if op == "+":
        v = av + bv
    elif op == "-":
        v = av - bv
        if v <= 0:
            return None
    elif op == "*":
        v = av * bv
    else:  # "/"
        if bv == 0 or av % bv != 0:
            return None
        v = av // bv
    return (v, f"({ae} {op} {be})")


def generate_problem(n, rng, num_range=20, target_max=999):
    """Generate a solvable Countdown problem with n numbers.
    Returns (nums, target, solution_expr) or None on a bad draw (caller retries)."""
    nums = [rng.randint(1, num_range) for _ in range(n)]
    work = [(x, str(x)) for x in nums]
    while len(work) > 1:
        i, j = rng.sample(range(len(work)), 2)
        a, b = work[i], work[j]
        ops = ["+", "-", "*", "/"]
        rng.shuffle(ops)
        res = None
        for op in ops:
            res = _combine(a, b, op)
            if res is not None and 1 <= res[0] <= target_max * 4:
                break
            res = None
        if res is None:
            return None
        work = [w for k, w in enumerate(work) if k not in (i, j)] + [res]
    target, expr = work[0]
    if not (1 <= target <= target_max):
        return None
    return nums, target, expr


def build_row(nums, target, expr):
    gt = {"target": int(target), "numbers": [int(x) for x in nums]}
    assert compute_score(f"<answer>{expr}</answer>", gt) == 1.0, "self-check failed"
    prompt = _PROMPT_TMPL.format(nums=" ".join(str(x) for x in nums), target=target)
    return {"prompt": prompt, "target": int(target), "nums": [int(x) for x in nums],
            "ground_truth": gt, "n_numbers": len(nums), "solution": expr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--difficulties", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--n-per-difficulty", type=int, default=150)
    ap.add_argument("--num-range", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-parquet", default=None,
                    help="Parquet whose problems to EXCLUDE (e.g. the train set, so test is disjoint).")
    ap.add_argument("--push-to-hub", default=None, help="Optional HF dataset repo id.")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    def _key(nums, target):
        return (int(target), tuple(sorted(int(x) for x in nums)))

    seen = set()  # dedup intra-set AND against --exclude-parquet (no train/test leakage)
    if args.exclude_parquet:
        import pandas as pd
        ex = pd.read_parquet(args.exclude_parquet)
        seen = {_key(n, t) for n, t in zip(ex["nums"], ex["target"])}
        print(f"[gen-hard] excluding {len(seen)} problems from {args.exclude_parquet}")

    rows = []
    for n in args.difficulties:
        made, tries = 0, 0
        while made < args.n_per_difficulty and tries < args.n_per_difficulty * 400:
            tries += 1
            p = generate_problem(n, rng, args.num_range)
            if p is None:
                continue
            k = _key(p[0], p[1])
            if k in seen:
                continue
            try:
                rows.append(build_row(*p))
                seen.add(k)
                made += 1
            except AssertionError:
                continue
        print(f"[gen-hard] n={n}: {made} problems ({tries} tries)")

    import pandas as pd
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[gen-hard] wrote {len(df)} problems -> {args.out}")
    print(f"[gen-hard] difficulty mix: {df['n_numbers'].value_counts().sort_index().to_dict()}")

    if args.push_to_hub:
        from datasets import Dataset
        Dataset.from_pandas(df).push_to_hub(args.push_to_hub, split="test")
        print(f"[gen-hard] pushed to hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
