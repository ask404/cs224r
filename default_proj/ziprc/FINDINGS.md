# ZIP-RC-Lite on Countdown — Findings & How We Got There

A retrospective on the stretch extension: findings, the engineering journey, and the
bugs that shaped it. The structured report with all tables is in `PROGRESS.md`; this
document is the narrative.

---

## 1. What we set out to build

The stretch extension from the proposal: **ZIP-RC-Lite** (Manvi et al. 2025) — give the
frozen RLOO Qwen2.5-0.5B policy *zero-overhead introspection*. A tiny head reads unused
vocabulary logits and, at every token, predicts a **joint distribution over (reward
outcome, remaining length)**. Those predictions then **allocate test-time compute
adaptively**. We also tried to specialize it to a **structured 3-outcome verifier**
(incoherent / coherent / correct).

Pipeline: `gen_rollouts → label (verifier) → train_head_only (frozen backbone) → score →
value_select → adaptive_decode (live) → adaptive_k → calibration_tv → figures`, on Modal.

---

## 2. How we got there

**Read the paper *and* the reference repo.** The biggest early wins came from the actual
reference code:
- The "structured 3-outcome verifier" is a **config flag** (`--reward-values 0.0 0.1 1.0`),
  not new architecture — the reference already supports arbitrary reward states.
- The repo ships **only offline scoring**; the *adaptive sampler* (the headline) had **no
  reference**, so we built it from the paper's math. Calibration = low-risk port; the live
  decoder = the hard novel part.
- Qwen2.5-0.5B has **271 unused logit slots** (ids 151665+) — free reserved tokens, no
  embedding resize needed.

**Bite-size milestones with a validation gate at each.** M0 (pure-Python units) → M1
(reserved-logit plumbing) → M2/M3 (verifier + occupancy audit) → M4/M5 (head training) →
**M6 keystone** (can the value head rank rollouts offline?) → M7/M8 (live decoder). The
expensive live-decoder build was gated behind a near-free offline test.

**The bugs, and the discipline that caught them.**

| Bug | How it surfaced | Fix |
|---|---|---|
| **Tied embeddings (the big one)** | Decoder generated coherent-but-degenerate text (`<loom>` not `<think>`); offline AUC looked *fine* (0.99) and **masked** it | 0.5B ties input/output embeddings → head-only training corrupted the *input* embeddings. Untie + clone the head. |
| **Wrong stop token** | Every sample ran to the 1024 cap into garbage (`latency=1024`) | Chat model ends on `<\|im_end\|>`, not `<\|endoftext\|>` |
| **Python 3.11 vs 3.13** | A PEP-701 f-string passed locally, `SyntaxError` on Modal | Added `tests/compile_py311.sh` pre-flight gate |
| **β-saturation** | The Pareto "frontier" was flat — β didn't bite | With K=8 redundancy, marginal value ≈0.004; β range was ~100× too high. Swept 0.002–0.1. |
| **aggregate dir** | Overnight pipeline step failed | Tried to save a PNG into a dir created by a later step; `mkdir` fix |

The recurring lesson, now baked into the repo: **offline metrics on clean rollouts can look
perfect while generation is broken** — so we validated the head's *generation quality*
against the raw policy, not just its scoring.

**Build out, then scale.** Once the decoder was validated, Stage C added the latency axis
(early-stop) + adaptive-K. An overnight detached run then scaled the head to 512 prompts,
added multi-seed error bars, the K=64 TV calibration, and cross-policy transfer.

---

## 3. Findings

### ① Calibration works, generalizes, and matches the paper
- Out-of-sample AUC(value→correct) = **0.92**; the *live* decoder's per-step readout matches
  the offline scorer → zero-overhead introspection works *during* generation.
- K=64 ground-truth calibration (the proposal's exact §3.4 metric): **TV 0.52 / F1 0.81 /
  acc 0.67** vs the paper's ZIP-RC-Lite **0.63 / 0.82 / 0.71** — we **match (beat on TV)**.

### ② Adaptive compute works — the policy flips with the objective (the α axis)
| regime | policy | result (multi-seed) |
|---|---|---|
| compute-bound | **prune** | **−63% compute**, −2.5 acc pts |
| latency-bound | **earlystop τ=0.8** | **−48% latency**, +1.7 acc |

Different policies win different objectives — the paper's compute-vs-latency story,
reproduced on Countdown.

### ③ Selection ≈ majority (honest scoping)
Value-*selection* doesn't beat majority voting on Countdown → the contribution is correctly
framed as adaptive *allocation*, not better answer selection.

### ④ Structured 3-outcome verifier = negative result + a real finding
Failure-mode diversity is **policy-capability-dependent**: strong RLOO fails *coherently*
(3% incoherent), weak SFT fails *incoherently* (10% coherent). No single 0.5B checkpoint
balances all three; forced on SFT, structured = binary (AUC 0.867 vs 0.865). This was the
proposal's pre-registered fallback.

### ⑤ Cross-policy transfer (exploratory) = policy-agnostic introspection
The RLOO-trained head scores SFT rollouts at 0.865 (≈ matched 0.867); the SFT head scores
RLOO rollouts at 0.890 (vs 0.922). The head learns a **transferable** "will this succeed"
signal.

### ⑥ Adaptive-K = null
On the scaled head, the small earlier gain didn't replicate — Countdown's narrow `value_first`
difficulty range leaves little to reallocate.

---

## 4. Honest caveats
- **Held-out is 50 prompts** (the test split's full size) — multi-seed error bars
  compensate; a bigger set would need carving from train.
- **Length head is weak** (E[remaining] MAE ≈ 257 tokens) — the *reward* half of the joint
  is well-calibrated, the *length* half isn't.
- **Single task / single base model** — findings are Countdown / Qwen2.5-0.5B specific.

---

## 5. Artifacts (`ask404/cs224r`, `main`)
- **`ziprc/`** — 18 modules + `PROGRESS.md` (structured report with all tables)
- **`tests/`** — 21 unit tests + the 3.11 pre-flight gate
- **`ziprc_results/figures_scaled/`** — calibration, selection, compute Pareto, latency
  Pareto, multi-seed error-bar Pareto, adaptive-K
- **`ziprc_results/*.json`** — raw metrics (TV calibration, aggregates)
- **PR #1** — the reviewable extension diff
- **Total compute spend: ~$25 of the $400 Modal budget.**

---

## 6. One-line summary

A complete, validated, scaled, and externally-validated ZIP-RC-Lite implementation: one
solid positive (adaptive compute saves ~63% compute or ~48% latency), one honest negative
(structured verifier is policy-capability-dependent), one bonus generalization result
(cross-policy transfer), and an engineering trail documenting every bug and the validation
discipline that caught it.
