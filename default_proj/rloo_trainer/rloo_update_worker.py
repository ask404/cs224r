"""Ray actor that applies policy-gradient updates for RLOO.

The orchestrator (`rloo.py`) samples responses and computes rewards, then
calls this worker with tokenized sequences to perform gradient updates.

This file is intentionally incomplete. Students are expected to implement
`update(...)` while reusing the data/model/sampling setup provided here.
"""

import os
from random import sample
from tokenize import group
import warnings
import ray
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

@ray.remote(num_gpus=1)
class RLOOUpdateWorker:
    """Owns policy/ref models and optimizer state for RLOO updates."""
    def __init__(
        self, 
        model_path, 
        optimizer_path, 
        scheduler_path,
        tokenizer_path=None, 
        ref_model_path=None,
        batch_size=64,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        group_size=16, 
        entropy_coefficient=0.01, 
        kl_divergence_coefficient=0.0, 
        lr_schedule='constant',
        learning_rate=1e-5, 
        weight_decay=0.01, 
        warmup_ratio=0.0,
        num_training_steps=250,
    ):
        self.model_path = model_path
        self.ref_model_path = ref_model_path if ref_model_path is not None else model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.group_size = group_size
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for RLOO, got {self.group_size}")
        self.entropy_coefficient = entropy_coefficient
        self.kl_divergence_coefficient = kl_divergence_coefficient
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        if warmup_ratio > 0:
            raise NotImplementedError("Warmup ratio > 0 is not supported for constant learning rate schedule")
        self.num_training_steps = num_training_steps

    def tear_down(self):
        """Release model/optimizer objects and clear GPU memory."""
        import gc
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ref_model'):
            del self.ref_model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_checkpoint_paths(self, model_path, optimizer_path, scheduler_path, load_checkpoint=False):
        """Update output paths (and optionally reload state immediately)."""
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        if load_checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load policy model, optional reference model, and optimizer/scheduler."""
        self.tear_down()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")
        self.model.gradient_checkpointing_enable()

        if self.kl_divergence_coefficient > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.ref_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device="cuda")
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        if self.optimizer_path and self.scheduler_path and os.path.exists(self.optimizer_path) and os.path.exists(self.scheduler_path):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")
            
            self.scheduler.load_state_dict(torch.load(self.scheduler_path))
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")

        self.model.train()

    def save_checkpoint(self):
        """Persist optimizer/scheduler state plus model+tokenizer weights."""
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        torch.save(self.scheduler.state_dict(), self.scheduler_path)

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    ## grad norm calculation 
    def _grad_norm(self):
        total = torch.zeros([], device="cuda")
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().float().pow(2).sum()
        return total.sqrt()

    def update_gradient_accumulation(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        device='cuda',
    ):
        """Split incoming batch into microbatches and call `update(...)`."""
        update_metrics = None
        if self.gradient_accumulation_steps > 1:
            curr_batch_size = input_ids.shape[0]
            assert curr_batch_size % self.gradient_accumulation_steps == 0, (
                f"Flattened batch size {curr_batch_size} must be divisible by gradient_accumulation_steps "
                f"{self.gradient_accumulation_steps}."
            )
            group_per_gradient_accumulation_step = curr_batch_size // self.gradient_accumulation_steps
            # Ensure each microbatch still contains full RLOO groups so the baseline is meaningful
            assert group_per_gradient_accumulation_step % self.group_size == 0, (
                f"Microbatch size {group_per_gradient_accumulation_step} must be divisible by group_size {self.group_size} "
                f"when using gradient_accumulation_steps={self.gradient_accumulation_steps}."
            )
            all_metrics = []
            for i in range(self.gradient_accumulation_steps):
                curr_input_ids = input_ids[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_attention_mask = attention_mask[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_is_response_token = is_response_token[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_rewards = rewards[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_sample_log_probs = None
                if sample_log_probs is not None:
                    curr_sample_log_probs = sample_log_probs[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                
                is_update_step = (i == self.gradient_accumulation_steps - 1)
                curr_update_metrics = self.update(
                    curr_input_ids,
                    curr_attention_mask,
                    curr_is_response_token,
                    curr_rewards,
                    curr_sample_log_probs,
                    is_update_step,
                    device,
                )
                all_metrics.append(curr_update_metrics)
            update_metrics = {}
            for metric_name in all_metrics[0].keys():
                update_metrics[metric_name] = np.mean([metric[metric_name] for metric in all_metrics]).item()
        else:
            update_metrics = self.update(
                input_ids,
                attention_mask,
                is_response_token,
                rewards,
                sample_log_probs,
                True,
                device,
            )

        return update_metrics

    # `is_update_step` is False on intermediate microbatches so we can
    # accumulate gradients before stepping optimizer/scheduler.
    def update(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        is_update_step: bool = True,
        device='cuda',
    ):
        # TODO(student): implement one RLOO policy update.
        # Inputs arrive flattened as [batch_size * group_size, seq_len].

        ## SETUP 

        # Convert numpy arrays to torch tensors on the same device as the model 
        input_ids = torch.as_tensor(input_ids, device=device, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, device=device)
        is_response_token = torch.as_tensor(is_response_token, device=device, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, device=device, dtype=torch.float32)
        if sample_log_probs is not None:
            sample_log_probs = torch.as_tensor(sample_log_probs, device=device, dtype=torch.float32)
    
        flat_batch_size, seq_len = input_ids.shape
        group_size = self.group_size

        # shape checks
        assert attention_mask.shape == input_ids.shape
        assert is_response_token.shape == input_ids.shape
        assert rewards.shape == (flat_batch_size,)
        assert flat_batch_size % group_size == 0
        num_prompts = flat_batch_size // group_size

        # single forward pass
        outputs = self.model (
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits 
              
        # Note: logits at position t predict token t+1 - shift logits and mask
        shifted_logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        target_attention_mask = attention_mask[:, 1:]
        target_response_mask = is_response_token[:, 1:]

        response_mask = target_attention_mask * target_response_mask

        # shape checks
        assert shifted_logits.shape[:2] == target_ids.shape
        assert target_ids.shape == response_mask.shape

# 1) Compute per-token log-probs on target tokens under current policy.
        log_probs = F.log_softmax(shifted_logits, dim=-1) 
        
        chosen_token_log_probs = log_probs.gather (
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        response_token_log_probs = chosen_token_log_probs * response_mask
        sequence_log_probs = response_token_log_probs.sum(dim=-1)
        
        assert chosen_token_log_probs.shape == target_ids.shape
        assert sequence_log_probs.shape == (flat_batch_size,)
        
        # response_lengths = response_mask.sum(dim=-1)
        # assert response_lengths.shape == (flat_batch_size,)

# 2) Build leave-one-out baseline within each response group.
        rewards_grouped = rewards.view(num_prompts, group_size)

        group_reward_sum = rewards_grouped.sum(dim=1, keepdim=True)
        rloo_baseline = (group_reward_sum - rewards_grouped) / (group_size - 1)

        advantages = rewards_grouped - rloo_baseline
        
        assert rewards_grouped.shape == (num_prompts, group_size)
        assert rloo_baseline.shape == (num_prompts, group_size)
        

# 3) Compute policy-gradient loss using advantages (and importance weights
        #    if sample_log_probs are provided).

        advantages_flat = advantages.reshape(flat_batch_size).detach()
        assert advantages_flat.shape == sequence_log_probs.shape

        
        ## importance weights
        if sample_log_probs is not None:
            assert sample_log_probs.shape == sequence_log_probs.shape

            log_ratio = sequence_log_probs.detach() - sample_log_probs
            log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
            importance_weights = torch.exp(log_ratio)
        
        else:
            importance_weights = torch.ones_like(sequence_log_probs)

      

# 4a) Add entropy regularization and optional KL penalty to ref model.
        probs = torch.exp(log_probs)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)
        masked_entropy = entropy_per_token * response_mask
        entropy = masked_entropy.sum() / response_mask.sum().clamp_min(1.0)

# 4b) Add optional KL penalty to ref model.
        if self.kl_divergence_coefficient > 0:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # KL(current || ref) per token
            kl_per_token = (
                probs * (log_probs - ref_log_probs)
            ).sum(dim=-1)

            kl_loss = (
                kl_per_token * response_mask
            ).sum() / response_mask.sum().clamp_min(1.0)
        else:
            kl_loss = torch.tensor(0.0, device=device)

        ### RLOO policy-gradient loss
        # Note: sequence_log_probs is the log prob of each sampled response 
        # Note: advantages_flat = reward - baseline 

        ## clipped importance weights
        importance_weights_for_loss = importance_weights.clamp(0.8, 1.2)
        policy_loss = - (importance_weights_for_loss * sequence_log_probs * advantages_flat).mean()
        loss = (
            policy_loss 
            - self.entropy_coefficient * entropy
            + self.kl_divergence_coefficient * kl_loss
        )

# 5) Backward pass; if `is_update_step`, clip and step optimizer/scheduler.
        assert torch.isfinite(loss) #check
        ## scale for gradient accumulation
        loss_for_backward = loss / self.gradient_accumulation_steps
        loss_for_backward.backward()


        if is_update_step:
            if self.gradient_clipping is not None and self.gradient_clipping > 0:  ## allows for grad clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clipping,
                )
            else: 
                grad_norm = self._grad_norm(device=device)


            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        
        else:
            grad_norm = torch.tensor(0.0, device=device)

     
       
# 6) Return scalar metrics used by trainer logging.
        if sample_log_probs is not None:
            log_ratio_raw = sequence_log_probs.detach() - sample_log_probs
        else:
            log_ratio_raw = torch.zeros_like(sequence_log_probs)


        return {
            "loss": loss.detach().float().item(),
            "policy_loss": policy_loss.detach().float().item(),
            "rloo_loss": policy_loss.detach().float().item(),
            "reward_mean": rewards.detach().float().mean().item(),
            "advantage_mean": advantages_flat.detach().float().mean().item(),
            "sequence_logprob_mean": sequence_log_probs.detach().float().mean().item(),
            "importance_weight_mean": importance_weights.detach().float().mean().item(),
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": float(grad_norm.detach().float().item()),
            "entropy": entropy.detach().float().item(),
            "kl_loss": kl_loss.detach().float().item(),
            "log_ratio_mean": log_ratio_raw.detach().float().mean().item(),
            "log_ratio_std": log_ratio_raw.detach().float().std().item(),
            "log_ratio_max": log_ratio_raw.detach().float().max().item(),
            "log_ratio_min": log_ratio_raw.detach().float().min().item(),
            "importance_weight_max": importance_weights.detach().float().max().item()
        }
        # raise NotImplementedError("This function is not implemented")


# love is a choice we get to make and it's a decision I plan to make every day
# 