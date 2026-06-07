#!/usr/bin/env python3
"""M8 (Stage A): live ZIP-RC-Lite adaptive parallel decoder.

This is the part with NO upstream reference (the repo ships only offline scoring).
Stage A builds the foundation and validates it before any meta-policy is trusted:

  * a manual K-parallel KV-cache decode loop on the head model;
  * read the reserved-slice joint at EVERY step (live, in the same forward), then
    MASK those logits before sampling so reserved tokens are never emitted;
  * a `prune` meta-policy: after a warmup, abandon the lowest-value samples (they
    were going to fail) and keep generating the rest.

Cost is measured as the paper does: total active forward passes (sum of active
samples over steps). Pruning is implemented by masking out abandoned samples and
only counting active ones, which yields a faithful cost-accuracy curve without KV
surgery (real KV-row deletion is a later FLOP optimization, same Pareto numbers).

Built-in validation (printed every run):
  - NO reserved token id ever appears in any generated sequence (masking works);
  - AUC(live value_end -> correct) ~ matches the offline scorer (readout is correct).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.countdown import compute_score  # noqa: E402
from ziprc.config import (DISTRIBUTION_TOKEN_ID, LENGTH_BINS_COUNTDOWN,
                          REWARD_VALUES_BINARY)  # noqa: E402


def _sample(logits, temperature, top_p, top_k):
    """logits: [B, V] (reserved already masked to -inf) -> next token [B]."""
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1).cumsum(-1)
        remove = probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


def _smooth(hist, w):
    if not hist:
        return 0.0
    tail = hist[-w:]
    return sum(tail) / len(tail)


class AdaptiveDecoder:
    def __init__(self, model, tok, start, reward_values, length_bins, device):
        self.model = model
        self.tok = tok
        self.start = start
        self.R = len(reward_values)
        self.L = len(length_bins) - 1
        self.num_bins = self.R * self.L
        self.reward_values = torch.tensor(reward_values, dtype=torch.float32, device=device)
        # Length-bin midpoints -> E[remaining tokens] from the LENGTH marginal. This is
        # the cost half of ZIP-RC's joint reward-cost signal (Stage B utility uses it).
        mids = [0.5 * (length_bins[i] + length_bins[i + 1]) for i in range(self.L)]
        self.length_mid = torch.tensor(mids, dtype=torch.float32, device=device)
        self.device = device
        self.eos = tok.eos_token_id
        # Chat models end the turn with <|im_end|>, NOT <|endoftext|>. Stop on either,
        # else samples run past their natural end into degenerate text to the token cap.
        self.stop_ids = {self.eos}
        for t in ("<|im_end|>", "<|endoftext|>"):
            tid = tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                self.stop_ids.add(tid)

    @torch.inference_mode()
    def decode_prompt(self, prompt, K, max_new_tokens, temperature, top_p, top_k,
                      policy="none", warmup=128, prune_interval=32, keep_min=2,
                      smooth_w=4, beta=0.0, stop_threshold=1.0, prune_threshold=0.5):
        tok, dev = self.tok, self.device
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        Lp = enc.input_ids.shape[1]
        input_ids = enc.input_ids.repeat(K, 1)
        attn = torch.ones_like(input_ids)

        out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        past = out.past_key_values

        done = [False] * K        # finished (wrote </answer> or eos)
        pruned = [False] * K       # abandoned by the meta-policy
        gen = [[] for _ in range(K)]
        vhist = [[] for _ in range(K)]      # predicted value (P(correct)) trajectory
        rhist = [[] for _ in range(K)]      # predicted E[remaining tokens] trajectory
        cost = 0                   # active forward passes (the Pareto cost metric)
        steps = 0

        def active(i):
            return not done[i] and not pruned[i]

        last_logits = out.logits[:, -1, :].float()
        for step in range(max_new_tokens):
            n_active = sum(active(i) for i in range(K))
            if n_active == 0:
                break
            steps += 1
            cost += n_active

            logits = last_logits
            joint = logits[:, self.start:self.start + self.num_bins].softmax(-1).view(K, self.R, self.L)
            val = (joint.sum(-1) @ self.reward_values)   # [K] reward marginal -> E[reward]
            rem = (joint.sum(1) @ self.length_mid)        # [K] length marginal -> E[remaining tokens]
            for i in range(K):
                if active(i):
                    vhist[i].append(float(val[i]))
                    rhist[i].append(float(rem[i]))

            logits = logits.clone()
            logits[:, self.start:self.start + self.num_bins] = float("-inf")
            nxt = _sample(logits, temperature, top_p, top_k)  # [K]

            for i in range(K):
                if not active(i):
                    continue
                t = int(nxt[i])
                gen[i].append(t)
                text = tok.decode(gen[i])
                if t in self.stop_ids or "</answer>" in text:
                    done[i] = True

            # meta-policy: prune lowest-value active samples after warmup
            if policy == "prune" and step >= warmup and step % prune_interval == 0:
                act = [i for i in range(K) if active(i)]
                if len(act) > keep_min:
                    act.sort(key=lambda i: _smooth(vhist[i], smooth_w))
                    for i in act[:len(act) - keep_min]:
                        if _smooth(vhist[i], smooth_w) < prune_threshold:  # only prune predicted-losers
                            pruned[i] = True

            # ZIP-RC utility policy: keep sample j iff its marginal contribution to
            # E[max reward] exceeds beta * its marginal remaining cost. Uses BOTH marginals.
            #   marginal value of j = v_j * prod_{i!=j}(1 - v_i)  (gain to best-of-set)
            #   marginal cost  of j = E[remaining tokens]_j / max_new_tokens
            # beta sweeps the Pareto frontier (beta=0 -> keep all; larger -> prune more).
            elif policy == "utility" and step >= warmup and step % prune_interval == 0:
                act = [i for i in range(K) if active(i)]
                if len(act) > keep_min:
                    vv = {i: min(0.999, max(1e-6, _smooth(vhist[i], smooth_w))) for i in act}
                    rr = {i: (rhist[i][-1] if rhist[i] else 0.0) for i in act}
                    total_log_fail = sum(math.log(1.0 - vv[i]) for i in act)
                    util = []
                    for j in act:
                        prod_others = math.exp(total_log_fail - math.log(1.0 - vv[j]))
                        delta = vv[j] * prod_others
                        cost_norm = rr[j] / max(1, max_new_tokens)
                        util.append((j, delta - beta * cost_norm))
                    util.sort(key=lambda x: x[1])  # ascending
                    for j, u in util[:len(act) - keep_min]:
                        if u < 0:
                            pruned[j] = True

            # LATENCY-bound (Stage C1): commit to the confident sample once its predicted
            # value crosses tau, abandoning the rest. We then stop as soon as IT finishes
            # instead of waiting for stragglers -> lower latency (and compute). Sweeping tau
            # traces the latency-accuracy frontier (the paper's alpha=0.1 regime).
            elif policy == "earlystop" and step >= warmup and step % prune_interval == 0:
                act = [i for i in range(K) if active(i)]
                if len(act) > 1:
                    best = max(act, key=lambda i: _smooth(vhist[i], smooth_w))
                    if _smooth(vhist[best], smooth_w) >= stop_threshold:
                        for i in act:
                            if i != best:
                                pruned[i] = True  # commit to the confident trajectory

            step_in = nxt.unsqueeze(1)
            attn = torch.cat([attn, torch.ones(K, 1, dtype=attn.dtype, device=dev)], dim=1)
            out = self.model(input_ids=step_in, attention_mask=attn,
                             past_key_values=past, use_cache=True)
            past = out.past_key_values
            last_logits = out.logits[:, -1, :].float()

        samples = []
        for i in range(K):
            text = tok.decode(gen[i])
            samples.append({
                "tokens": gen[i],
                "text": text,
                "finished": done[i],
                "pruned": pruned[i],
                "n_tokens": len(gen[i]),
                "vhist": list(vhist[i]),     # full predicted-value trajectory (for offline prune sim)
                "value_end": (sum(vhist[i][-16:]) / len(vhist[i][-16:])) if vhist[i] else float("nan"),
                "value_first": vhist[i][0] if vhist[i] else float("nan"),
            })
        return {"samples": samples, "cost": cost, "latency": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Trained head-only model dir.")
    ap.add_argument("--dataset", default="asingh15/countdown_tasks_3to4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--distribution-token-id", type=int, default=DISTRIBUTION_TOKEN_ID)
    ap.add_argument("--reward-values", type=float, nargs="+", default=list(REWARD_VALUES_BINARY))
    ap.add_argument("--length-bins", type=int, nargs="+", default=list(LENGTH_BINS_COUNTDOWN))
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--prune-interval", type=int, default=32)
    ap.add_argument("--keep-min", type=int, default=2)
    ap.add_argument("--betas", type=float, nargs="*", default=[],
                    help="If set, run the ZIP-RC utility policy at each beta -> compute Pareto.")
    ap.add_argument("--stop-thresholds", type=float, nargs="*", default=[],
                    help="If set, run the latency-bound earlystop policy at each tau -> latency Pareto.")
    ap.add_argument("--out-parquet", default=None)
    ap.add_argument("--pareto-out", default=None, help="Write per-config Pareto summary JSON.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = args.distribution_token_id
    L = len(args.length_bins) - 1
    num_bins = len(args.reward_values) * L

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to(device).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(device).eval()
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    assert start + num_bins <= lm_head.weight.size(0)

    dec = AdaptiveDecoder(model, tok, start, args.reward_values, args.length_bins, device)

    ds = load_dataset(args.dataset, split=args.split).select(range(args.num_prompts))
    prompts = list(ds["prompt"])
    targets = list(ds["target"]) if "target" in ds.column_names else [g["target"] for g in ds["ground_truth"]]
    nums = list(ds["nums"]) if "nums" in ds.column_names else [g["numbers"] for g in ds["ground_truth"]]

    # Configs: none + prune + ZIP-RC utility (compute axis, beta) + earlystop (latency axis, tau).
    configs = [("none", {}), ("prune", {})]
    configs += [("utility", {"beta": b}) for b in args.betas]
    configs += [("earlystop", {"stop_threshold": t}) for t in args.stop_thresholds]
    results = {}
    rows = []
    reserved_violation = 0
    for pi, prompt in enumerate(prompts):
        gt = {"target": int(targets[pi]), "numbers": [int(x) for x in nums[pi]]}
        for policy, params in configs:
            if policy == "utility":
                key = f"util b={params['beta']:g}"
            elif policy == "earlystop":
                key = f"estop t={params['stop_threshold']:g}"
            else:
                key = policy
            torch.manual_seed(args.seed + pi)  # same seed per prompt -> fair policy comparison
            r = dec.decode_prompt(prompt, args.K, args.max_new_tokens, args.temperature,
                                  args.top_p, args.top_k, policy=policy,
                                  warmup=args.warmup, prune_interval=args.prune_interval,
                                  keep_min=args.keep_min, **params)
            comp = [s for s in r["samples"] if s["finished"] and not s["pruned"]]
            pool = comp if comp else [s for s in r["samples"] if not s["pruned"]] or r["samples"]
            sel = max(pool, key=lambda s: s["value_end"])
            sel_correct = compute_score(sel["text"], gt) == 1.0
            any_correct = any(compute_score(s["text"], gt) == 1.0 for s in r["samples"])
            d = results.setdefault(key, {"acc": [], "oracle": [], "cost": [], "latency": []})
            d["acc"].append(float(sel_correct))
            d["oracle"].append(float(any_correct))
            d["cost"].append(r["cost"])
            d["latency"].append(r["latency"])
            for s in r["samples"]:
                for t in s["tokens"]:
                    if start <= t < start + num_bins:
                        reserved_violation += 1
                if key == "none":
                    rows.append({"prompt_idx": pi, "value_end": s["value_end"],
                                 "value_first": s["value_first"],
                                 "correct": float(compute_score(s["text"], gt) == 1.0)})

    print("\n=== STAGE-A VALIDATION ===")
    print(f"  reserved-token emissions (must be 0): {reserved_violation}")
    rdf = pd.DataFrame(rows)
    if len(rdf) and rdf["correct"].nunique() > 1:
        def _auc(lab, sc):
            lab, sc = np.asarray(lab, float), np.asarray(sc, float)
            o = np.argsort(sc); rk = np.empty_like(o, float); rk[o] = np.arange(1, len(sc) + 1)
            npos, nneg = lab.sum(), (1 - lab).sum()
            return float((rk[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
        print(f"  live AUC(value_end -> correct):  {_auc(rdf['correct'], rdf['value_end']):.3f}  (offline was ~0.99)")
        print(f"  live AUC(value_first -> correct):{_auc(rdf['correct'], rdf['value_first']):.3f}  (offline was ~0.81)")

    print("\n=== POLICY / PARETO (cost = active forward passes; lower-left is better) ===")
    base_cost = np.mean(results["none"]["cost"])
    print(f"  {'config':<12} {'acc':>6} {'oracle':>7} {'cost':>8} {'latency':>8}  vs-none")
    for key in results:
        acc = np.mean(results[key]["acc"])
        orc = np.mean(results[key]["oracle"])
        cost = np.mean(results[key]["cost"])
        lat = np.mean(results[key]["latency"])
        save = "baseline" if key == "none" else f"{(1 - cost / base_cost) * 100:+.1f}% cost"
        print(f"  {key:<12} {acc:6.3f} {orc:7.3f} {cost:8.0f} {lat:8.1f}  {save}")

    if args.pareto_out:
        import json
        summary = {key: {"acc": float(np.mean(results[key]["acc"])),
                         "oracle": float(np.mean(results[key]["oracle"])),
                         "cost": float(np.mean(results[key]["cost"])),
                         "latency": float(np.mean(results[key]["latency"]))}
                   for key in results}
        with open(args.pareto_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[decode] wrote Pareto summary -> {args.pareto_out}")

    if args.out_parquet and len(rdf):
        rdf.to_parquet(args.out_parquet, index=False)
        print(f"\n[decode] wrote per-sample values -> {args.out_parquet}")


if __name__ == "__main__":
    main()
