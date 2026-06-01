# ZIP-RC-Lite for Countdown — Progress Report

**Extension:** zero-overhead introspective prediction (ZIP-RC-Lite, Manvi et al. 2025)
specialized to a structured verifier, on the frozen RLOO Qwen2.5-0.5B policy.

## Summary

We implemented and validated the stretch extension end-to-end: a frozen-backbone
ZIP-RC head that predicts a joint distribution over (reward outcome, remaining length)
from ~20 unused vocabulary logits at zero inference overhead, plus a **live adaptive
parallel decoder** that uses those predictions to allocate test-time compute. The core
result holds out-of-sample, and the cost–accuracy Pareto frontier is complete.

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

**2. Adaptive compute works (headline).** Cost–accuracy Pareto (cost = active forward
passes; oracle@8 = 0.825 ceiling):

| config | acc | cost | vs none |
|---|---|---|---|
| none (full BoN) | 0.700 | 4072 | — |
| util β=0.005 | 0.700 | 3084 | −24% |
| util β=0.01 | 0.675 | 2429 | −40% |
| util β=0.02 | 0.650 | 1703 | −58% |
| util β=0.05 | 0.600 | 1152 | −72% |
| **prune** | **0.700** | **1250** | **−69%** |

The ZIP head enables ~69% compute reduction at **zero accuracy loss**. The β-sweep gives
a smooth tunable frontier; notably the simple absolute-threshold **prune dominates** the
redundancy-aware utility policy on this task.

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

Stage C (branching + latency-bound α for the latency-vs-compute axis); intermediate-capability
checkpoints to test whether a balanced 3-class regime exists; the structured failure-mode
signal may be more useful to the curriculum half than to inference.
