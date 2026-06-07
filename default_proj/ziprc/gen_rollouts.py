#!/usr/bin/env python3
"""M1: generate on-policy Countdown rollouts for ZIP-RC head training.

Single-process vLLM generator (no multiprocessing DP) sized for one GPU and the
0.5B policy. Feeds the dataset's already-chat-formatted `prompt` strings directly
(NO chat template re-application, unlike the upstream Qwen3 generator), stops on
</answer> like the project's SamplingWorker, and writes a parquet whose schema
matches the vendored ZIP-RC dataset/scorer:

  prompt_idx, prompt, target, nums, ground_truth,
  prompt_token_ids, output_token_ids, input_ids, label_positions,
  response, length, finished, finish_reason

Because ZIP-RC-Lite freezes the backbone, you generate here with the PLAIN policy
checkpoint (no head, no logit masking needed); scoring later reads identical
hidden states.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ziprc.alignment import make_label_positions  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="asingh15/qwen-sft-countdown-defaultproj",
                   help="Policy checkpoint to roll out (HF repo or local path).")
    p.add_argument("--tokenizer", default=None, help="Defaults to --model.")
    p.add_argument("--dataset", default="asingh15/countdown_tasks_3to4")
    p.add_argument("--from-parquet", default=None,
                   help="Read prompts from a local/volume parquet (prompt,target,nums) instead of --dataset.")
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--max-num-prompts", type=int, default=200)
    p.add_argument("--samples-per-prompt", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    tok_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    if args.from_parquet:
        df = pd.read_parquet(args.from_parquet).iloc[: args.max_num_prompts]
        prompts = list(df["prompt"])
        targets = list(df["target"])
        nums = list(df["nums"])
    else:
        ds = load_dataset(args.dataset, split=args.split)
        ds = ds.select(range(min(args.max_num_prompts, len(ds))))
        prompts = list(ds["prompt"])
        # Countdown eval schema: target + nums. Fall back to ground_truth if present.
        targets = list(ds["target"]) if "target" in ds.column_names else [g["target"] for g in ds["ground_truth"]]
        nums = list(ds["nums"]) if "nums" in ds.column_names else [g["numbers"] for g in ds["ground_truth"]]

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print(f"[gen] {len(prompts)} prompts x {args.samples_per_prompt} samples from {args.model}", flush=True)
    outputs = llm.generate(prompts, sampling)

    rows = []
    for i, out in enumerate(outputs):
        prompt_token_ids = list(out.prompt_token_ids)
        gt = {"target": int(targets[i]), "numbers": [int(x) for x in nums[i]]}
        for o in out.outputs:
            output_token_ids = list(o.token_ids)
            input_ids, label_positions = make_label_positions(prompt_token_ids, output_token_ids)
            finish_reason = getattr(o, "finish_reason", None)
            rows.append({
                "prompt_idx": i,
                "prompt": prompts[i],
                "target": gt["target"],
                "nums": gt["numbers"],
                "ground_truth": gt,
                "prompt_token_ids": prompt_token_ids,
                "output_token_ids": output_token_ids,
                "input_ids": input_ids,
                "label_positions": label_positions,
                "response": o.text,
                "length": len(output_token_ids),
                "finished": finish_reason in ("stop", "length", "eos"),
                "finish_reason": finish_reason,
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, engine="pyarrow", index=False)
    print(f"[gen] wrote {len(df)} rollouts -> {args.out}", flush=True)
    print(f"[gen] mean response length: {df['length'].mean():.1f} tokens "
          f"(p50={df['length'].median():.0f}, p95={df['length'].quantile(0.95):.0f})", flush=True)


if __name__ == "__main__":
    main()
