# Experiment: does adaptive-K help when difficulty *varies*? (harder Countdown)

A concrete, ready-to-run test of the hypothesis from `ADAPTIVE_K.md`: **adaptive-K's gain
scales with the variance of prompt difficulty.** Countdown-3to4 is too narrow (`value_first`
range ≈ [0.52, 0.54], adaptive-K ≈ null). We widen difficulty *within the same task family*
using a controlled knob — **operand cardinality** — and re-measure.

Generator: `make_countdown_hard.py` (validated; every problem self-checked to
`compute_score == 1.0`). Difficulty label = `n_numbers ∈ {3,4,5,6}` (more numbers ⇒ larger
search ⇒ harder), derivable at analysis time as `len(nums)`.

## Why operand cardinality
It's a clean, monotone, *measurable* difficulty axis that keeps the task, prompt format,
reward, and base policy fixed — so any change in adaptive-K's gain is attributable to
difficulty *spread*, not a confound. n=3 is near-trivial for the RLOO policy; n=6 needs a
deep combination it often misses ⇒ a wide, labeled difficulty distribution.

## Step 0 — build the dataset (CPU, ~seconds)
```bash
python ziprc/make_countdown_hard.py --out data/hard_train.parquet --n-per-difficulty 200 --seed 0
python ziprc/make_countdown_hard.py --out data/hard_test.parquet  --n-per-difficulty 60  --seed 1
# push as a single HF dataset with train/test splits (so gen_rollouts can load it):
python -c "from datasets import Dataset, DatasetDict; import pandas as pd; \
DatasetDict({'train': Dataset.from_pandas(pd.read_parquet('data/hard_train.parquet')), \
'test': Dataset.from_pandas(pd.read_parquet('data/hard_test.parquet'))}).push_to_hub('ask404/countdown-hard-mixed')"
```

## Steps 1–5 — reuse the EXISTING pipeline, just point `--dataset` at the hard set
```bash
DS=ask404/countdown-hard-mixed
# 1) rollouts from the frozen RLOO policy (train for the head, test for eval)
modal run ziprc_modal.py gen   -- --model $POLICY --dataset $DS --split train --out /vol/ziprc/data/hard_train_rollouts.parquet --max-num-prompts 800 --samples-per-prompt 4
modal run ziprc_modal.py gen   -- --model $POLICY --dataset $DS --split test  --out /vol/ziprc/data/hard_test_rollouts.parquet  --max-num-prompts 240 --samples-per-prompt 8
# 2) label (binary correctness is deterministic; heuristic judge is free)
modal run ziprc_modal.py label -- --in-parquet /vol/ziprc/data/hard_train_rollouts.parquet --out-parquet /vol/ziprc/data/hard_train_labeled.parquet --judge heuristic
modal run ziprc_modal.py label -- --in-parquet /vol/ziprc/data/hard_test_rollouts.parquet  --out-parquet /vol/ziprc/data/hard_test_labeled.parquet  --judge heuristic
# 3) train the ZIP-RC head (same config as the main runs)
modal run ziprc_modal.py train -- --model-id $POLICY --data-path /vol/ziprc/data/hard_train_labeled.parquet --weights-path /vol/ziprc/models/lite_binary_hard --label-column correct --reward-values 0.0 1.0 --batch-size 16 --gradient-accumulation-steps 2 --num-epochs 3
# 4) score the held-out hard rollouts
modal run ziprc_modal.py score -- --model /vol/ziprc/models/lite_binary_hard --in-parquet /vol/ziprc/data/hard_test_labeled.parquet --out-parquet /vol/ziprc/data/hard_test_scored.parquet --reward-values 0.0 1.0
# 5) adaptive-K
modal run ziprc_modal.py adaptive_k -- --scored /vol/ziprc/data/hard_test_scored.parquet --budgets 2 3 4 5 6 --kmax 8 --trials 16 --out-json /vol/ziprc/data/adaptive_k_hard.json
```
Note: a wider sample budget (`--samples-per-prompt`/`--kmax` up to 16) makes the
hard tier's pass@k less saturated and gives adaptive-K more to allocate.

## Metrics (and the predictions that would confirm the hypothesis)
1. **Difficulty signal range** — `value_first` min/median/max. *Predict:* spreads well
   beyond Countdown-3to4's ≈ [0.52, 0.54] (this is the precondition for any gain).
2. **Per-tier pass@1** by `n_numbers`. *Predict:* monotone drop n=3 → n=6 (confirms real
   difficulty variance; the easy tier near-saturated, the hard tail far from it).
3. **adaptive-K gain** = `hardness − fixed` oracle@meanK. *Predict:* now **positive and
   replicating** (vs the Countdown null), strongest at *medium* budgets where extra samples
   flip frontier prompts.
4. **Gain vs spread** — repeat on `{3,4}` (narrow) vs `{3,4,5,6}` (wide) subsets and plot
   the gain against the empirical difficulty variance. *Predict:* monotone — the cleanest
   single result, directly testing the hypothesis.

## Minor analysis add-on
`adaptive_k.py` already takes the scored parquet; stratifying by `len(nums)` (one `groupby`)
gives per-tier pass@1 and per-tier gain. The dataset also carries `n_numbers` if you prefer
an explicit label. No pipeline change is needed — difficulty is recoverable from `nums`.

## Cost
Tiny: generation is CPU/seconds; rollouts + one head train + offline scoring ≈ the cost of a
single scaleup leg (~\$3–5). All steps are validated code paths.
