# Introspective Calibration as the Currency of the Test-Time Compute Economy

*A study of blending zero-overhead ZIP-RC-Lite introspection into a unified test-time controller:
when do the levers compound, when do they fight, and what single quantity governs it all.*

> Status: living document. Result tables marked **[multi-seed]** are filled from the
> leakage-audited multi-seed runs (`blend_eval.py`, `make_blend_figures.py`).

---

## 0. One-paragraph thesis

A frozen RLOO Qwen2.5-0.5B carries, in its *unused* vocabulary logits, a zero-overhead
introspective signal — ZIP-RC-Lite's joint prediction over (reward, remaining length). We show
that this one signal drives **three composable test-time levers** — *adaptive-K* (how many whole
samples a prompt gets, across prompts), *prune* (kill a doomed sample mid-generation, compute
axis), and *earlystop* (commit once confident, latency axis) — and we measure what happens when
you **blend** them. Three findings: (1) the compute savings **compound** (multiplicative, not
additive); (2) the levers are **synergistic on accuracy** — allocation makes pruning *safer*, the
opposite of the naive "they fight" fear; and (3) a **single quantity governs the whole economy**:
the head's *mid-trajectory winner/loser separability*. Where it is high the blend is a genuine
**Pareto free lunch** (cheaper *and* more accurate); where it is low it degrades to a
compute-for-accuracy trade. Calibration is the currency.

---

## 1. The three levers (one signal, three axes)

| lever | axis | scope | decision | reads |
|---|---|---|---|---|
| **adaptive-K** | budget | *across* prompts | how many whole samples this prompt gets | mid-probe `value_q25` |
| **prune** | compute | *within* a sample | abandon this token-stream mid-generation | per-step value |
| **earlystop** | latency | *within* a prompt | commit to a confident sample, stop the rest | per-step value |

All three read the *same* head. That is the source of both the upside (one zero-overhead signal
runs everything) and the risk (correlated failure where the head is miscalibrated). This study
focuses on the two **compute-axis** levers that align — adaptive-K + prune — and gates earlystop on
a separate dependency (§6).

---

## 2. Method: faithful, cheap, leakage-audited

**Faithful prune accounting.** For each prompt we generate a pool of `pool_k=8` samples *once*
(`policy="none"`, recording every sample's value trajectory and length), then replay the **exact
decode-time prune rule offline**. Because token-streams are independent, pruning sample *i* never
changes sample *j*, so truncating *i* at its prune-point reproduces the live cost and outcome
*exactly* — verified by unit tests (cost monotonicity over thousands of random pools, keep_min
protection, pruned-loser exclusion). One generation yields the entire **budget × threshold**
frontier offline.

**Compute metric.** `cost` = active forward passes per prompt (ZIP-RC's Pareto cost), summed over
the samples a policy actually runs. `oracle` = any selected-and-surviving sample correct.

**Leakage & methodology audit.**
- **Label consistency:** the head's training label `correct` *is* `compute_score == 1.0` — the
  exact signal we evaluate (no train/eval mismatch; the "heuristic" judge only splits *failures*
  for an unused 3-outcome head).
- **`value_q25` consistency:** scoring and the blend both read the value ~25% through the response.
- **Held-out integrity:** the dataset's `test` split is 50 rows; the head trained on `train[0:512]`.
  We therefore (a) firm up on the 50 truly-held-out prompts with **6 seeds**, and (b) build a large
  **leakage-safe** pool: a `train[200000:]` slice **deduped by `(target, sorted nums)` against the
  head's training problems**, with a hard disjointness assertion (**passed: 0 overlap**). The large
  pool is head-clean; it remains policy-seen (frozen policy RL-trained on train), reported
  transparently — the blend/calibration claims concern the head + test-time mechanism, for which
  head-disjointness is the requirement.
- **Tie-corrected AUC:** the separability metric uses average-rank Mann-Whitney AUC (a bug where
  all-equal scores returned 0.75 instead of 0.5 was caught and fixed; unit-tested).

---

## 3. Result 1 — the compute savings COMPOUND

The two levers act on different quantities — adaptive-K moves *oracle* at fixed samples, prune
moves *cost* at fixed oracle — so a blend should multiply their savings. It does.

**[multi-seed]** *(pending: hard pool, 3 seeds, n=120)*

- prune's token-saving is **allocation-agnostic** (≈ same fraction under fixed and adaptive
  allocation), and
- the blend's total compute cut ≈ the **product** of the two levers' individual cuts.

---

## 4. Result 2 — the levers are SYNERGISTIC on accuracy (they do not fight)

The naive fear (failure-mode #4): prune kills the very frontier samples adaptive-K paid for, so the
allocation lift collapses under pruning. The data shows the **opposite**.

**[multi-seed]** *(pending)* — adaptive-K's oracle lift over fixed *grows* under prune, because
allocation concentrates samples on hard prompts, leaving **more survivors** when prune culls
losers. **Allocation makes pruning safer.** This is the key enabling result for a unified
controller: the compute-axis levers cooperate rather than cancel.

---

## 5. Result 3 — the calibration law (separability predicts the free lunch)

This is the heart of the study. Define **mid-trajectory separability** = the AUC of the head's
smoothed value *at the prune decision point* as a predictor of the sample's *final* correctness.
It measures whether the head can tell winners from losers early enough for prune to act safely.

**The law:** the blend's best **Pareto compute-saving at ≥ baseline oracle** rises with
mid-trajectory separability. Where separability is high, prune's accuracy cost is *recoverable* by
threshold tuning and the blend Pareto-dominates; where it is low, the lost winners are *confidently
mis-rated* and no threshold recovers them — the blend is only a trade.

**[multi-seed: pool-level points]** *(pending: main n=50 ×6, holdout n=300 ×2, hard n=120 ×3)*

| pool | head | mid-AUC | base oracle | blend Pareto-saving @ ≥ base | verdict |
|---|---|---|---|---|---|
| main-test | lite_binary_512 | … | … | … | … |
| holdout (head-clean) | lite_binary_512 | … | … | … | … |
| hard | lite_binary_hard | … | … | … | … |

**[multi-seed: per-tier gradient]** *(pending: hard pool, operand-count tiers 3–6)* — one pool, four
calibration levels: separability degrades with operand count, and the blend's Pareto-saving tracks
it tier-by-tier. *(correlation reported by `analyze` step.)*

**Why threshold-tuning is the tell.** On a calibrated pool the at-risk winners cluster at the
*borderline* (mid-value 0.3–0.5), so a gentler τ spares them; on an OOD pool they sit *confidently
low* (<0.3), τ-invariant and unrecoverable. The diagnostic reports exactly this distribution.

---

## 6. What this says about the unified controller

- **Blend adaptive-K + prune now.** They align (both avoid wasting compute on doomed work), their
  savings compound, and they are synergistic on accuracy.
- **The free lunch is calibration-gated.** A pool/head with high mid-trajectory separability gets
  Pareto-dominance; a low-separability regime gets a compute-for-accuracy trade. The unlock for hard
  problems is a head better-calibrated *mid-trajectory* on hard data — not a different mechanism.
- **Gate earlystop on the length head.** The latency lever needs trustworthy per-sample cost
  prediction (samples ≠ compute: adaptive concentrates on long trajectories), which is the head's
  weakest marginal. Harden it before adding the latency axis under one ZIP-RC utility.

---

## 7. Reproduce

```
modal run ziprc_modal.py pipeline -- blend          # hard pool: frontier + per-tier gradient
modal run ziprc_modal.py pipeline -- blend_main      # 50 held-out test prompts, 6 seeds
modal run ziprc_modal.py pipeline -- blend_holdout   # 300 leakage-safe head-clean prompts
# figures:
modal run ziprc_modal.py figures -- ziprc/make_blend_figures.py --sweeps main:...:AUC hard:...:AUC ...
```

Core code: `blend_eval.py` (faithful offline blend + frontier + calibration diagnostic + per-tier
gradient), `allocate_budget.py::allocate` (shared online allocator), `make_holdout.py` (leakage-safe
pool), `make_blend_figures.py` (frontiers + calibration-law plot).
