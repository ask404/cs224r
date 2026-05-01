"""Starter SFT training entrypoint for the class project.

This file is intentionally incomplete. Students are expected to implement
`train(...)` while reusing the data/model setup provided here.
"""

import sys
from pathlib import Path

# Allow `python sft_trainer/sft.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from sft_trainer.sft_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device='cuda', use_gradient_checkpointing=True):
    """Load policy model + tokenizer for SFT training."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()
    return model, tokenizer

def clear_cache(model):
    """Best-effort GPU/CPU cache cleanup between heavy steps."""
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir):
    """Save model/tokenizer plus optimizer/scheduler states."""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

    torch.save({
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(output_dir, 'train_states.pth'))
    print(f"Model saved to {output_dir}")

def train(
    model, 
    tokenizer, 
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0
):
    # TODO(student): implement the SFT optimization loop.
    # Expected high-level flow:
    # 1) Forward pass on `input_ids` and compute token-level log-probs.
    # 2) Mask loss to response tokens only using `is_response_token`.
    # 3) Backprop, optionally clip gradients, then optimizer/scheduler steps.
    # 4) Periodically evaluate on `test_dataloader` and log metrics to W&B.
    # 5) Save checkpoints under `output_dir` when requested.

    global_step = 0
    num_batches = len(train_dataloader)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_tokens = 0.0

        train_correct_tokens = 0.0
        train_total_tokens = 0.0

        progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            response_token = batch["is_response_token"].to(device).bool() & attention_mask.bool()

            # 1) Forward pass on `input_ids` and compute token-level log-probs.
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [batch size, number of tokens, vocabulary size]

            # 2) Mask loss to response tokens only using `is_response_token`.
            # Reference: https://sebastianraschka.com/faq/docs/next-token-prediction.html
            shift_logits = logits[:, :-1, :] # Prediction for next tokens.
            shift_labels = input_ids[:, 1:] # Next tokens targets.
            shift_mask = response_token[:, 1:].float() # Align 'is_response_token' with shift_labels.

            num_tokens = shift_mask.sum()

            # Required to log Token Accuracy on Train.
            pred_tokens = shift_logits.argmax(dim=-1) # Choose max token number from vocabulary.
            correct_tokens = ((pred_tokens == shift_labels)*shift_mask).sum()
            train_correct_tokens += correct_tokens.detach().item()
            train_total_tokens += num_tokens.detach().item()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            tokens_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [batch size, num. tokens - 1]

            tokens_losses = -(tokens_logp * shift_mask)
            loss_sum = tokens_losses.sum()
            batch_loss = loss_sum / num_tokens.clamp_min(1.0)

            epoch_loss += loss_sum.detach().item()
            epoch_tokens += num_tokens.detach().item()

            # 3) Backprop, optionally clip gradients, then optimizer/scheduler steps.
            loss_grad_accum = batch_loss / gradient_accumulation_steps
            loss_grad_accum.backward()
            do_step= (((step + 1) % gradient_accumulation_steps) == 0) or ((step + 1) == num_batches)

            if do_step:
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                optimizer.step()
                scheduler.step() # Updates the lr schedule.
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = epoch_loss / max(epoch_tokens, 1.0)
                lr = scheduler.get_last_lr()[0]

                train_acc = train_correct_tokens / max(train_total_tokens, 1.0)

                wandb.log({
                    "train/loss": avg_loss,
                    "train/batch_loss": batch_loss.item(),
                    "train/token_accuracy": train_acc,
                    "train/lr": lr,
                    "train/epoch": epoch + 1,
                }, step=global_step)

                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

        # 4) Periodically evaluate on `test_dataloader` and log metrics to W&B.
        model.eval()
        test_loss = 0.0
        test_tokens = 0.0
        test_correct_tokens = 0.0

        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader, desc="Evaluating", leave=True):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                response_token = batch["is_response_token"].to(device).bool() & attention_mask.bool()

                prediction_out = model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = prediction_out.logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                shift_mask = response_token[:, 1:].float()

                # Required to log Token Accuracy on Test.
                pred_tokens = shift_logits.argmax(dim=-1)  # Choose max token number from vocabulary.
                correct_tokens = ((pred_tokens == shift_labels) * shift_mask).sum()
                test_correct_tokens += correct_tokens.detach().item()

                log_probs = F.log_softmax(shift_logits, dim=-1)
                tokens_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

                test_loss += (-(tokens_logp * shift_mask)).sum().item()
                test_tokens += shift_mask.sum().item()

        train_avg = epoch_loss / max(epoch_tokens, 1.0)
        test_avg = test_loss / max(test_tokens, 1.0)
        test_acc = test_correct_tokens / max(test_tokens, 1.0)

        wandb.log({
            "eval/loss": test_avg,
            "eval/token_accuracy": test_acc,
            "eval/epoch": epoch + 1,
        }, step=global_step)

        print(f"Epoch {epoch + 1}/{num_epochs} | train_loss={train_avg:.4f} | eval_loss={test_avg:.4f}")

        clear_cache(model)

        # 5) Save checkpoints under `output_dir` when requested.
        if save_model:
            ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}")
            save_checkpoint(model, tokenizer, optimizer, scheduler, ckpt_dir)

    if save_model:
        save_checkpoint(model, tokenizer, optimizer, scheduler,
                        os.path.join(output_dir, "final"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='Asap7772/cog_behav_all_strategies')
    parser.add_argument('--output_dir', type=str, default='sft_model')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_response_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='sft_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

    dataloaders = get_dataloaders(
        dataset_name=args.dataset_name, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length, 
        max_response_length=args.max_response_length, 
        batch_size=args.batch_size, 
        splits=['train', 'test'],
        pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']
    # Scheduler steps happen only after an optimizer step, so account for
    # gradient accumulation when estimating total training steps.
    num_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model, 
        tokenizer, 
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping
    )

if __name__ == "__main__":
    main()
