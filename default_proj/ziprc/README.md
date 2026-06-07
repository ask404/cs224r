# ZIP-RC-Lite for Countdown (stretch extension)

Frozen-backbone ZIP-RC head (Manvi et al. 2025) specialized to a **structured
3-outcome verifier** {incoherent, coherent, correct} on Countdown / Qwen2.5-0.5B.

Adapted from the official repo: https://github.com/rohinmanvi/ZIP-RC
(`ziprc_dataset.py`, `train_head_only.py`, `score_joint_head.py` are derivatives;
the upstream repo ships no LICENSE file — used here for a class project with
attribution; cite the paper).

## Findings & writeups
- **[`BLEND_STUDY.md`](BLEND_STUDY.md)** — *flagship.* Blending adaptive-K + prune into a
  test-time controller, with a full rigor layer (prompt bootstrap, held-out τ selection,
  difficulty-controlled partial correlation, leakage + faithfulness audits). Honest headline:
  ~20–25% compute saving at neutral accuracy; savings super-compound; prune-safety is governed by
  mid-trajectory separability (set by the *problem* more than the head).
- **[`ADAPTIVE_K.md`](ADAPTIVE_K.md)** — the adaptive-K arc (start-based fails → mid-trajectory
  works → cap-headroom governs the gain), and the first blend sketch (§4c).
- **[`FINDINGS.md`](FINDINGS.md)**, **[`CROSS_POLICY.md`](CROSS_POLICY.md)**,
  **[`EXPERIMENT_difficulty.md`](EXPERIMENT_difficulty.md)** — calibration, cross-policy transfer,
  difficulty stratification.

## Pipeline (each stage persists to the Modal volume; stages are isolated)

```
gen_rollouts.py   policy rollouts (vLLM, plain ckpt)      -> rollouts.parquet
label_rollouts.py compute_score + coherence judge + AUDIT -> +correct, +reward3
train_head_only.py freeze backbone, train reserved logits -> head model dir
score_joint_head.py teacher-forced read of reserved slice -> +value_mean, +value_end
value_select.py   KEYSTONE viability: value@K vs random/oracle
```

## Run it

**M0 unit tests (local, no GPU):**
```bash
cd default_proj && PYTHONPATH=. python3 tests/test_grid.py && \
  PYTHONPATH=. python3 tests/test_alignment.py && \
  PYTHONPATH=. python3 tests/test_verifier.py && \
  PYTHONPATH=. python3 tests/test_reserved_ids.py
```

**Tiny viability smoke (Modal, binary, NO API key):**
```bash
export HF_TOKEN=...; export ZIPRC_GPU=A10G
bash ziprc/smoke_countdown.sh
```

**Structured (3-outcome) path** — same chain, swap the label step to Haiku and the
head to `reward3`:
```bash
export ANTHROPIC_API_KEY=...            # judge runs on failures only, cached
modal run ziprc_modal.py label -- --in-parquet .../rollouts.parquet \
  --out-parquet .../labeled.parquet --judge haiku --judge-cache /vol/ziprc/data/judge_cache.json
modal run ziprc_modal.py train -- --data-path .../labeled.parquet \
  --weights-path /vol/ziprc/models/lite_struct --label-column reward3   # reward-values default 0 0.1 1.0
modal run ziprc_modal.py score -- --model /vol/ziprc/models/lite_struct \
  --in-parquet .../labeled.parquet --out-parquet .../scored_struct.parquet
modal run ziprc_modal.py select -- --in-parquet .../scored_struct.parquet
```

## Key config (`ziprc/config.py`) — and why it differs from upstream

| Constant | Value | Why |
|---|---|---|
| `DISTRIBUTION_TOKEN_ID` | **151665** | Qwen2.5-0.5B's first unused logit (upstream 151669 is a *Qwen3* id). 271 free slots. |
| `LENGTH_BINS_COUNTDOWN` | `[0,16,…,1024]` | Upstream's `[0,256,…,32768]` is for 32k math traces; Countdown is <~300 tok → all mass in bin 0. **Re-audit (M3) and tune.** |
| `REWARD_VALUES_STRUCTURED` | `[0.0,0.1,1.0]` | incoherent / coherent / correct. Binary baseline = `[0.0,1.0]`. |

## Gotchas baked into the code

- **No gradient checkpointing** in `train_head_only.py`: backbone is frozen, no grad
  flows through it, so checkpointing would only waste compute.
- **No logit masking offline**: frozen backbone ⇒ generation (plain ckpt) and scoring
  (head ckpt) share identical hidden states; masking is only needed for a *live* decoder.
- **M3 gate** (printed by `label_rollouts.py`): each structured class ≥15% and length
  bins non-degenerate. If the gate fails, fall back to the binary head and report it.
- **Viability gate** (`value_select.py`): if `value@K ≈ random@K`, the head can't rank
  rollouts — stop before building the live adaptive sampler.

## What's NOT here yet (by design)

The **live adaptive meta-action sampler** (prune/stop/branch → the cost–accuracy Pareto
curve) is the high-risk piece with no upstream reference. It is gated behind the
`value_select.py` viability read above. Offline value-selection already gives a real
cost–accuracy comparison (≈ Weighted Best-of-N) without it.
