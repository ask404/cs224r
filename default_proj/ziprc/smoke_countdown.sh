#!/usr/bin/env bash
# Tiny end-to-end ZIP-RC-Lite viability run on Modal (BINARY head, NO API key).
# Proves the whole chain works and gives a first viability read in one go.
#
#   export HF_TOKEN=...            # to pull the policy checkpoint
#   export WANDB_API_KEY=...       # optional
#   export ZIPRC_GPU=A10G          # cheap; override if you want speed (L4/A100-40GB)
#   bash ziprc/smoke_countdown.sh
#
# ~24 prompts x 8 samples. Expect a few $ and <20 min. If value@8 > random@8 and
# trends toward oracle@8, the head ranks rollouts -> proceed to structured + Haiku.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

POLICY="${ZIPRC_POLICY:-asingh15/qwen-sft-countdown-defaultproj}"   # swap to your RLOO ckpt
D=/vol/ziprc/data
M=/vol/ziprc/models
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-3600}"

echo "### 1/5 generate rollouts (GPU)"
modal run ziprc_modal.py gen -- \
  --model "$POLICY" --out "$D/smoke_rollouts.parquet" \
  --max-num-prompts 24 --samples-per-prompt 8 --max-tokens 1024

echo "### 2/5 label + M3 audit (CPU, heuristic judge, no API)"
modal run ziprc_modal.py label -- \
  --in-parquet "$D/smoke_rollouts.parquet" --out-parquet "$D/smoke_labeled.parquet" \
  --judge heuristic

echo "### 3/5 train BINARY head-only (GPU)"
modal run ziprc_modal.py train -- \
  --model-id "$POLICY" --data-path "$D/smoke_labeled.parquet" \
  --weights-path "$M/smoke_lite_binary" \
  --label-column correct --reward-values 0.0 1.0 --max-steps 150

echo "### 4/5 score rollouts with head (GPU)"
modal run ziprc_modal.py score -- \
  --model "$M/smoke_lite_binary" --in-parquet "$D/smoke_labeled.parquet" \
  --out-parquet "$D/smoke_scored.parquet" --reward-values 0.0 1.0

echo "### 5/5 VIABILITY: value-based selection vs random/oracle (CPU)"
modal run ziprc_modal.py select -- --in-parquet "$D/smoke_scored.parquet" --ks 1 2 4 8

echo "### done. read the value@K vs random@K / oracle@K table above."
