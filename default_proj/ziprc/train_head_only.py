#!/usr/bin/env python3
"""M4/M5: ZIP-RC-Lite — freeze backbone, train only the LM head to predict the
joint P(reward_state, tokens_remaining) over the reserved logit slice.

Adapted from github.com/rohinmanvi/ZIP-RC (src/train_ziprc_head_only.py). Changes
for Countdown / Qwen2.5-0.5B:
  - distribution_token_id default 151665 (Qwen2.5 reserved slot; NOT the Qwen3 151669).
  - length_bins + reward_values flow through to ZIPDataset (short Countdown bins).
  - NO gradient checkpointing: the backbone is frozen and the loss is computed from
    output_hidden_states via a manual F.linear over the head, so NO gradient flows
    through the transformer body — checkpointing it would only burn compute.
  - Single-process friendly (DDP only engages under torchrun/RANK).
  - Visualization import is optional.

No KL term: the frozen backbone cannot drift, so the next-token policy is preserved
by construction (this is the whole point of the Lite variant).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ziprc.config import LENGTH_BINS_COUNTDOWN, REWARD_VALUES_STRUCTURED, assert_slice_fits  # noqa: E402
from ziprc.ziprc_dataset import ZIPDataset  # noqa: E402

try:
    import wandb
except Exception:
    wandb = None


def compute_loss(model, batch, distribution_token_id, num_bins):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # [B, S, E]

    tgt = model.module if hasattr(model, "module") else model
    lm_module = getattr(tgt, "_orig_mod", tgt)
    lm_head = lm_module.lm_head if hasattr(lm_module, "lm_head") else lm_module.get_output_embeddings()

    all_b, all_pos, all_labels = [], [], []
    for i, (pos_list, label_list) in enumerate(zip(batch["label_positions"], batch["bin_labels"])):
        for p, l in zip(pos_list, label_list):
            if 0 <= p < hidden_states.size(1):
                all_b.append(i)
                all_pos.append(p)
                all_labels.append(l)
    if not all_pos:
        z = torch.tensor(0.0, device=device)
        return z, z

    b_idx = torch.as_tensor(all_b, device=device, dtype=torch.long)
    s_idx = torch.as_tensor(all_pos, device=device, dtype=torch.long)
    labels = torch.as_tensor(all_labels, device=device, dtype=torch.long)
    h_flat = hidden_states[b_idx, s_idx, :]

    weight_bins = lm_head.weight[distribution_token_id: distribution_token_id + num_bins]
    bias_bins = (lm_head.bias[distribution_token_id: distribution_token_id + num_bins]
                 if getattr(lm_head, "bias", None) is not None else None)
    logits_bins = F.linear(h_flat, weight_bins, bias_bins)
    loss = F.cross_entropy(logits_bins, labels, reduction="mean")
    return loss, loss


def untie_and_clone_lm_head(model):
    """Qwen2.5-0.5B has tie_word_embeddings=True, so lm_head.weight IS the input
    embedding. Training the head while tied leaks gradient into the INPUT embeddings
    (the loss flows through the backbone's token embeddings) and corrupts the policy,
    degrading generation. Untie + clone so the head is a SEPARATE parameter: only the
    output head trains; input embeddings stay frozen and generation is preserved.
    (Upstream's 1.5B/1.7B models are untied, so they never hit this.)"""
    if not getattr(model.config, "tie_word_embeddings", False):
        return False
    in_emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    has_bias = getattr(out, "bias", None) is not None
    new_head = torch.nn.Linear(out.in_features, out.out_features, bias=has_bias)
    new_head.weight = torch.nn.Parameter(in_emb.weight.detach().clone())
    if has_bias:
        new_head.bias = torch.nn.Parameter(out.bias.detach().clone())
    new_head = new_head.to(device=in_emb.weight.device, dtype=in_emb.weight.dtype)
    model.set_output_embeddings(new_head)
    model.config.tie_word_embeddings = False
    return True


def freeze_backbone_keep_lm_head_trainable(model):
    for p in model.parameters():
        p.requires_grad = False
    tgt = getattr(model, "_orig_mod", model)
    head = tgt.lm_head if hasattr(tgt, "lm_head") else tgt.get_output_embeddings()
    for p in head.parameters():
        p.requires_grad = True


def train(model, dataset, cfg, num_bins):
    distributed = int(os.environ.get("RANK", -1)) != -1
    if distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world, master = 1, True

    torch.manual_seed(42)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=(sampler is None),
                        sampler=sampler, collate_fn=ZIPDataset.collate_fn)

    if master and cfg.wandb_project and wandb is not None:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name or f"ziprc_lite_{int(time.time())}",
                   config={**vars(cfg), "num_bins": num_bins,
                           "num_reward_states": dataset.num_reward_states,
                           "num_length_bins": dataset.num_length_bins})

    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[device])
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    if master:
        print(f"[train] trainable params: {n_train} (head-only)", flush=True)

    optimizer = AdamW(trainable, lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    total_iters = max(1, (cfg.num_epochs * len(loader)) // cfg.gradient_accumulation_steps)
    warmup_iters = int(cfg.warmup_ratio * total_iters)

    def lr_at(i):
        if warmup_iters > 0 and i < warmup_iters:
            return cfg.learning_rate * i / warmup_iters
        if total_iters <= warmup_iters:
            return 0.0
        progress = (i - warmup_iters) / (total_iters - warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * progress)) * cfg.learning_rate

    global_step = 0
    accum = 0.0
    for epoch in range(cfg.num_epochs):
        if distributed:
            sampler.set_epoch(epoch)
        for it, batch in enumerate(loader):
            update = (it + 1) % cfg.gradient_accumulation_steps == 0
            if distributed and isinstance(model, DDP):
                model.require_backward_grad_sync = update
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                loss, _ = compute_loss(model, batch, cfg.distribution_token_id, batch["num_bins"])
                loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accum += loss.item()
            if not update:
                continue
            if cfg.grad_clip:
                clip_grad_norm_(trainable, cfg.grad_clip)
            global_step += 1
            lr = lr_at(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if master and global_step % cfg.log_every == 0:
                print(f"[train] step {global_step}/{total_iters} loss {accum:.4f} lr {lr:.2e}", flush=True)
                if cfg.wandb_project and wandb is not None:
                    wandb.log({"train/loss": accum, "lr": lr, "step": global_step})
            accum = 0.0
            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                break
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break

    if master:
        tgt = model.module if hasattr(model, "module") else model
        tgt = getattr(tgt, "_orig_mod", tgt)
        tgt.save_pretrained(cfg.weights_path)
        print(f"[train] saved head-only model -> {cfg.weights_path}", flush=True)
        if cfg.wandb_project and wandb is not None:
            wandb.finish()
    if distributed:
        dist.barrier()


def main_worker(local_rank, world_size, cfg):
    os.environ.update(WORLD_SIZE=str(world_size), RANK=str(local_rank),
                      LOCAL_RANK=str(local_rank), MASTER_ADDR=cfg.master_addr,
                      MASTER_PORT=str(cfg.master_port))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", trust_remote_code=True)
    except Exception as e:
        if local_rank == 0:
            print(f"[train] flash_attention_2 unavailable ({e}); default attention.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)

    model.config.use_cache = False
    # NOTE: deliberately NO gradient_checkpointing_enable() — backbone is frozen.
    if untie_and_clone_lm_head(model) and local_rank == 0:
        print("[train] untied tied embeddings -> only the output head trains "
              "(input embeddings frozen).", flush=True)
    freeze_backbone_keep_lm_head_trainable(model)

    dataset = ZIPDataset(cfg.data_path, max_length=cfg.max_length,
                         reward_values=cfg.reward_values, length_bins=cfg.length_bins,
                         label_column=cfg.label_column)
    num_bins = assert_slice_fits(dataset.reward_values, dataset.length_bins,
                                 start=cfg.distribution_token_id)
    assert num_bins == dataset.num_bins

    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        os.makedirs(cfg.weights_path, exist_ok=True)
        tokenizer.save_pretrained(cfg.weights_path)

    train(model, dataset, cfg, num_bins)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="asingh15/qwen-sft-countdown-defaultproj")
    p.add_argument("--data-path", required=True)
    p.add_argument("--weights-path", required=True)
    p.add_argument("--label-column", default="reward3", help="'reward3' (structured) or 'correct' (binary).")
    p.add_argument("--distribution-token-id", type=int, default=151665)
    p.add_argument("--reward-values", type=float, nargs="+", default=None,
                   help="Default: structured [0,0.1,1.0]. Binary: 0.0 1.0.")
    p.add_argument("--length-bins", type=int, nargs="+", default=None,
                   help="Remaining-token bin edges; default = short Countdown bins.")
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-name", default="")
    return p.parse_args()


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def main():
    cfg = parse_args()
    if cfg.reward_values is None:
        cfg.reward_values = list(REWARD_VALUES_STRUCTURED) if cfg.label_column != "correct" else [0.0, 1.0]
    if cfg.length_bins is None:
        cfg.length_bins = list(LENGTH_BINS_COUNTDOWN)
    cfg.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    cfg.master_port = int(os.environ.get("MASTER_PORT", str(_find_free_port())))
    ngpus = torch.cuda.device_count() or 1
    if ngpus == 1:
        main_worker(0, 1, cfg)  # avoid mp.spawn overhead on single GPU
    else:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
