# Cross-Policy Transfer — the introspection head is (mostly) policy-agnostic

A focused write-up of the exploratory positive finding: a ZIP-RC head trained on one
policy's rollouts predicts success on a *different* policy's rollouts with little or no loss.

Code path: train a head with `train_head_only.py` on policy A's labeled rollouts, then
`score_joint_head.py` + `value_select.py` on policy B's rollouts (mismatched head/data) — all
validated; no new code.

## Setup
Two post-training stages of the **same base** (Qwen2.5-0.5B): the weaker **SFT** policy
(pass@1 ≈ 0.20) and the stronger **RLOO** policy (pass@1 ≈ 0.66). We train a binary ZIP-RC
head on each and cross-evaluate `AUC(value → correct)` on each policy's held-out rollouts.

## Result

| head ↓ \ data → | RLOO rollouts | SFT rollouts |
|---|---|---|
| **RLOO head** | **0.922** (matched) | **0.865** (cross) |
| **SFT head**  | **0.890** (cross) | **0.867** (matched) |

- **RLOO head → SFT data: 0.865 vs the matched SFT head's 0.867 → essentially lossless.**
- **SFT head → RLOO data: 0.890 vs the matched RLOO head's 0.922 → only −0.032.**

Both off-diagonal transfers stay ≥ 0.865 — the head's "will this rollout succeed" signal is
**not tied to the policy it was trained on.**

## What it means
- **The head reads something general** ("is this trajectory on track"), not a policy's
  idiosyncrasies. ZIP-RC heads are normally treated as *model-specific* (the paper trains one
  per model); this shows they are robust to **policy drift** within a base model.
- **An informative asymmetry:** the *strong*-policy head transfers **down losslessly**, while
  the *weak*-policy head transfers **up** with a small drop. Intuition: the strong policy's
  rollouts are a **richer distribution** (more correct *and* more diverse failures), so a head
  trained on them generalizes better; the weak policy's rollouts are narrow (mostly incoherent
  failures), so its head is slightly less calibrated on the strong policy's diverse rollouts.
  (This mirrors the capability-dependence finding in `PROGRESS.md`.)

## Why it matters
1. **Train once, reuse across RL.** RL updates the policy continually, but you would *not*
   need to retrain the introspection head every step — a meaningful practical saving for any
   ZIP-RC-in-the-training-loop use.
2. **Success-prediction looks like a transferable capability**, which is the key enabler for
   the agentic directions (see the cross-references below): a single head as a portable
   confidence/escalation signal rather than a per-policy artifact.

## Honest caveats
- **Same base model, different post-training only.** This is transfer across SFT↔RLOO of
  *Qwen2.5-0.5B* — **not** across base models or architectures. The head still reads that
  base's hidden states via the reserved-logit mechanism, so cross-*base* transfer is a
  separate (and harder) question.
- Held-out sets are 50 prompts each — directionally clear (a 0.06 gap is well outside noise),
  but tighten with more prompts before headlining.

## Follow-on
1. **Transfer across intermediate RLOO checkpoints** (head from step 0 → policy at step 100):
   directly measures robustness to *RL drift* and validates "train once, reuse."
2. **Transfer across base sizes/families** (0.5B head → 1.5B policy, or Qwen → a different
   base): probably fails because hidden states differ, but a high-value negative — it bounds
   whether a "universal" introspection probe is even possible without retraining.
3. **Agentic use** (see the brainstorm in the chat / future `AGENT_DIRECTIONS.md`): a portable
   confidence signal for SLM-first routing/escalation — *call for help when self-predicted
   success is low* — exactly the CLEAR cost-aware regime.

## One-line summary
A ZIP-RC head trained on one policy predicts success on another policy's rollouts nearly as
well as a matched head (off-diagonal AUC ≥ 0.865 vs ≤ 0.922 matched), so introspection is
largely **policy-agnostic** within a base model — heads survive policy drift, with the
strong-policy head transferring down losslessly and the weak-policy head transferring up with
a small drop.
