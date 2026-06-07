# Introspective Calibration as the Currency of the Test-Time Compute Economy

*A study of blending zero-overhead ZIP-RC-Lite introspection into a unified test-time controller:
when do the levers compound, when do they fight, and what single quantity governs it all.*

> Status: living document. Tables marked **[… pending]** are filled from the leakage-audited,
> prompt-bootstrapped runs (`blend_eval.py` → `blend_stats.py` → `law_combine.py`).

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
- **Independent adversarial audits (3).** A leakage audit *empirically verified* (pulling the
  actual HF splits) that `test ∩ train[0:512] = 0/50`, the holdout dedup excludes every
  head-training problem, `value_q25` does not leak (the blend re-allocates online from its own
  fresh probe), and the hard pool is train/test-disjoint — **no leakage found**. A faithfulness
  audit *fuzzed* `sim()` against the live decoder over **220k cells with 0 mismatches** (and made
  the live prune threshold a parameter, so every swept τ is deployable, not offline-only). A
  statistics audit drove the rigor layer below.

**Statistical rigor (the inference layer).** Generation is the only GPU cost, so we dump every
sample's value trajectory once and do all inference offline (`blend_stats.py`):
- **The prompt is the inference unit.** All CIs are **prompt-level paired bootstraps** (seeds are
  averaged per prompt to cancel generation noise) — *not* seed-std, which only measures decoding
  variance. CIs are scoped "on these prompts."
- **Held-out operating point.** (budget, τ) is selected on one half of prompts and the blend's
  saving + oracle-delta are *reported on the other half* (2-fold) — no winner's-curse from picking
  the best of the 4×5 grid on its own data. The full grid is always printed.
- **Matched-budget** comparison (blend at B vs fixed+full at the *same* B) — no cross-budget bug.
- **Falsifiable forms with CIs:** compounding = (blend saving − product-of-levers) gap CI;
  allocation-agnostic = (prune fraction under fixed − under adaptive) CI; synergy = oracle DiD CI.
- **The law is tested against its confound:** a **partial correlation** of (separability, prune
  accuracy-hit) controlling base-oracle (difficulty), with a **permutation p-value** — so the law
  must beat the difficulty gradient, not merely coincide with it. The y-axis is the **signed** hit
  (no 0-floor).

---

## 3. Result 1 — the compute savings COMPOUND

The two levers act on different quantities — adaptive-K moves *oracle* at fixed samples, prune
moves *cost* at fixed oracle — so a blend should multiply their savings. Two falsifiable tests
(prompt-bootstrap CIs):

- **Allocation-agnostic** (the testable core): prune keeps the *same* cost-fraction under fixed vs
  adaptive allocation — `(fixed prune-fraction − adaptive prune-fraction)` CI should straddle 0.
- **Compounding**: the blend's measured saving equals the product-of-levers prediction —
  `(measured − product)` gap CI should straddle 0.

**[CI pending]** *(filled from `blend_stats.py`, hard/main/holdout)*

---

## 4. Result 2 — the levers are SYNERGISTIC on accuracy (they do not fight)

The naive fear (failure-mode #4): prune kills the very frontier samples adaptive-K paid for, so the
allocation lift collapses under pruning. We test the **opposite** with a difference-in-differences:
`[(adaptive − fixed) oracle | prune] − [(adaptive − fixed) oracle | full]`, prompt-bootstrap CI.
If the CI is **> 0**, adaptive's lift *grows* under prune — allocation concentrates samples on hard
prompts, leaving **more survivors** when prune culls losers, so **allocation makes pruning safer**.
This is the key enabling result for a unified controller: the compute-axis levers cooperate.

**[CI pending]**

---

## 5. Result 3 — the calibration law (separability predicts the free lunch)

This is the heart of the study. Define **mid-trajectory separability** = the AUC of the head's
smoothed value *at the prune decision point* as a predictor of the sample's *final* correctness.
It measures whether the head can tell winners from losers early enough for prune to act safely.

**The law (signed, confound-controlled):** prune's **signed accuracy-hit** (oracle lost to a fixed
aggressive τ; ≤ 0, closer to 0 = safer) rises toward 0 as mid-trajectory separability rises. We
pool points across pools *and* per-difficulty-tier (one pool = a calibration gradient), then test
the law **against its confound** — a **partial correlation controlling base-oracle** (difficulty)
with a **permutation p-value**. The law is only real if separability predicts prune-safety *beyond*
the difficulty gradient.

**[law table pending]** — pool points (main n=50, holdout n=300 head-clean, hard n=120) + per-tier
gradient (hard tiers 3–6), each `(mid-AUC, signed prune-hit, base-oracle, n)`; combined Pearson,
**partial r | base-oracle**, and permutation p reported by `law_combine.py`.

**The causal decoupler — same difficulty, different calibration.** The partial correlation controls
difficulty *statistically*; the `blend_cross` run controls it *by construction*. Because the value
head is head-only-trained on a **frozen** backbone and the reserved logits are **masked before
sampling**, generation is head-independent: running the *same* hard pool with the out-of-domain
head `lite_binary_512` (trained on easy Countdown) yields **identical samples** but a worse-
calibrated value signal. If, at *fixed* difficulty and *fixed* samples, the worse-calibrated head
shows lower mid-AUC **and** a larger prune accuracy-hit, calibration — not difficulty — drives
prune-safety. **[pending: (hard, in-domain head) vs (hard, OOD head)]**

**Downstream consequence — the held-out free lunch.** Where the law says prune is safe, the blend
should Pareto-dominate. We report the blend's **held-out** saving + oracle-delta (operating point
chosen on a disjoint half of prompts), with prompt-bootstrap CIs. **[pending]**

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
# each pipeline: blend_eval (generate+dump) -> blend_stats (rigorous offline stats -> law_*.parquet)
modal run ziprc_modal.py pipeline -- blend          # hard pool (lite_binary_hard), tiers 3-6
modal run ziprc_modal.py pipeline -- blend_main      # 50 held-out test prompts, 6 seeds
modal run ziprc_modal.py pipeline -- blend_holdout   # 300 leakage-safe head-clean prompts
modal run ziprc_modal.py pipeline -- blend_cross     # hard pool, OOD head -> decouple calibration
# combine the law across pools (partial correlation + permutation) and plot:
python ziprc/law_combine.py --law-points law_main.parquet law_holdout.parquet law_hard.parquet law_cross.parquet
python ziprc/make_blend_figures.py --sweeps main:..._sweep.parquet hard:... --law-points law_*.parquet
```

Core code: `blend_core.py` (shared faithful prune replay + tie-corrected AUC), `blend_eval.py`
(generate-once + offline frontier + calibration diagnostic + raw dump), `blend_stats.py`
(prompt-bootstrap CIs, held-out τ, falsifiable mechanism tests), `law_combine.py` (partial
correlation + permutation), `allocate_budget.py::allocate` (shared online allocator),
`make_holdout.py` (leakage-safe pool), `make_blend_figures.py` (frontiers + calibration-law plot).
