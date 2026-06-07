# ZIP-RC-Lite for Countdown — Progress Report

**Extension:** zero-overhead introspective prediction (ZIP-RC-Lite, Manvi et al. 2025)
specialized to a structured verifier, on the frozen RLOO Qwen2.5-0.5B policy.

## Summary

We implemented and validated the stretch extension end-to-end: a frozen-backbone
ZIP-RC head that predicts a joint distribution over (reward outcome, remaining length)
from ~20 unused vocabulary logits at zero inference overhead, plus a **live adaptive
parallel decoder** that uses those predictions to allocate test-time compute. The core
result holds out-of-sample, and the cost–accuracy Pareto frontier is complete.

**Follow-on — the blend study (`BLEND_STUDY.md`).** We blended the across-prompt allocator
(adaptive-K) with within-sample pruning and subjected it to a full rigor layer — prompt-level
bootstrap CIs, held-out operating-point selection, a difficulty-controlled partial correlation, and
two independent adversarial audits (leakage + faithfulness, both clean). Honest result: a **robust
~20–25% compute saving at neutral accuracy** whose savings **super-compound**; prune-safety is
governed by *mid-trajectory* separability (≈0.6–0.7 AUC, far below the 0.91 value-end AUC) as a
within-pool tendency. The rigor repeatedly **corrected the optimistic single-run claims** (the
"Pareto free lunch" was winner's-curse; the "accuracy synergy" didn't survive).

## What was built (`ziprc/`)

A six-stage Modal pipeline, all validated:
`gen_rollouts` → `label_rollouts` (verifier) → `train_head_only` (frozen backbone) →
`score_joint_head` (offline) → `value_select` → `adaptive_decode` (live), plus
`make_figures`. The head lives in reserved logit ids `151665+` of Qwen2.5-0.5B.

## Results (held-out: head trained on 256 prompts, evaluated on unseen prompts)

**1. Calibration generalizes.** AUC(value→correct) by prefix position:

| first | q25 | mean | end |
|---|---|---|---|
| 0.849 | 0.805 | 0.899 | 0.910 |

The live decoder's per-step readout matches the offline scorer (0.906 vs 0.910),
confirming the zero-overhead introspection works *during* generation.

**2. Adaptive compute works (headline).** The head's introspection drives three test-time
strategies, each winning a different regime (held-out, K=8; oracle@8 ≈ 0.81):

*Compute-bound* — `prune` / utility (β sweep), cost = active forward passes:

| config | acc | cost | vs none |
|---|---|---|---|
| none (full BoN) | 0.708 | 4141 | — |
| util β=0.01 | 0.688 | 2343 | −43% |
| util β=0.05 | 0.625 | 1160 | −72% |
| **prune** | **0.708** | **1266** | **−69%** |

*Latency-bound* — `earlystop` (τ sweep), cost = decode steps (wall-clock proxy):

| config | acc | latency | vs none |
|---|---|---|---|
| none | 0.708 | 688 | — |
| **estop τ=0.8** | **0.729** | **384** | **−44%** |
| estop τ=0.9 | 0.729 | 440 | −36% |

`prune` cuts ~69% of compute and `earlystop` ~44% of latency, both at **zero accuracy
loss** — and the winning policy flips with the objective (the paper's α axis). The simple
absolute-threshold `prune` dominates the redundancy-aware utility policy on this task.

**2b. Adaptive-K allocation.** Using the head's per-prompt difficulty signal (`value_first`,
AUC 0.85) to spend more samples on predicted-hard prompts beats fixed-K at matched average
budget by a small but consistent margin (+0.6–1.8 oracle points across mean-K 2–6), limited
by Countdown's narrow difficulty spread.

**3. Selection ≈ majority (honest scoping).** Value-based *selection* does not beat
majority voting on Countdown (value@8 = 0.74 vs majority@8 = 0.78 = oracle). The
contribution is correctly framed as adaptive compute *allocation* (ZIP-RC's thesis), not
better answer selection.

**4. Structured 3-outcome verifier — negative result (pre-registered).** Failure-mode
diversity is policy-capability-dependent: the converged RLOO policy fails almost only
*coherently* (3% incoherent), while the weaker SFT policy fails almost only
*incoherently* (10% coherent). No single 0.5B checkpoint populates all three classes
above 15%. Trained on SFT (all three present), the structured head matched binary
(AUC 0.867 vs 0.865) — the proposal's stated fallback.

## Scaled & extended results (512-prompt head, multi-seed; figures in `ziprc_results/figures_scaled/`)

**Scaled head.** Trained on 512 prompts, held-out calibration improves to **AUC 0.922**.

**Multi-seed error bars (3 seeds).** Tight std; a more honest headline than single-seed:

| config | acc | cost | latency |
|---|---|---|---|
| none | 0.708±0.012 | 3944±106 | 692±11 |
| **prune** | 0.683±0.012 | **1468±16** (−63%) | 508±16 |
| **estop τ=0.8** | **0.725±0.020** | 2306±147 | **356±12** (−48%) |

`prune` trades ~2.5 acc points for −63% compute; `earlystop` gives −48% latency at +1.7 acc.

**K=64 ground-truth calibration (proposal §3.4 metric) — validates against the paper:**

| metric | ours (ZIP-RC-Lite) | paper's ZIP-RC-Lite |
|---|---|---|
| mean Total Variation | **0.52** | ~0.63 |
| end-of-gen reward F1 | 0.81 | ~0.82 |
| end-of-gen reward acc | 0.67 | ~0.71 |

Our reward calibration matches (and beats on TV) the paper's reported ZIP-RC-Lite numbers.
The length head is weak (E[remaining-tokens] MAE ≈ 257 tokens) — an honest limitation: the
reward half of the joint is well-calibrated, the length half is not.

**Cross-policy transfer (exploratory).** Heads transfer across policies with minimal drop —
the RLOO-trained head scores SFT rollouts at AUC 0.865 (vs the matched SFT head's 0.867), and
the SFT head scores RLOO rollouts at 0.890 (vs matched 0.922). **Introspection is largely
policy-agnostic** — the head learns a transferable "will this rollout succeed" signal.

**Adaptive-K** did not replicate its small gain on the scaled head (≈0, within noise) — the
narrow `value_first` dynamic range on Countdown leaves little to reallocate.

## Engineering notes (non-obvious bugs, all caught by a validation harness)

- **Tied embeddings (critical):** Qwen2.5-0.5B ties input/output embeddings, so head-only
  training silently corrupted the policy's generation. Fixed by untying + cloning the head;
  upstream's larger (untied) models never hit this.
- **Stop token:** the chat model ends turns on `<|im_end|>`, not `<|endoftext|>`.
- **Python 3.11 vs 3.13:** Modal runs 3.11; a PEP-701 f-string passed locally (3.13) but
  failed remotely. Added `tests/compile_py311.sh` as a pre-flight gate.

## Cost & reproduction

~$15–20 of the $400 Modal budget (bite-size staging; most cost was one-time image builds).
Run order: `bash ziprc/smoke_countdown.sh` for the tiny binary path; see `ziprc/README.md`
for the full structured/held-out pipeline. Figures: `ziprc_results/figures/`.

## Possible next steps

A larger run (512-prompt head, 256 held-out) to tighten the numbers; true mid-generation
branching (needs per-sample KV caches; likely marginal on short Countdown sequences);
intermediate-capability checkpoints to test whether a balanced 3-class regime exists; the
structured failure-mode signal may be more useful to the curriculum half than to inference.
