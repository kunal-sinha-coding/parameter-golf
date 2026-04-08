#!/usr/bin/env bash
set -euo pipefail

# CPU debug variant of baseline_cpu.sh. It creates a tiny local dataset prefix
# from the existing shard files, then runs train_gpt_cpu.py against that debug
# dataset with a much smaller training budget.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
DEBUG_DATA_PATH="${DEBUG_DATA_PATH:-./data/datasets/fineweb10B_sp1024_cpu_debug}"
DEBUG_TRAIN_TOKENS="${DEBUG_TRAIN_TOKENS:-262144}"
DEBUG_VAL_TOKENS="${DEBUG_VAL_TOKENS:-131072}"

python3 data/make_debug_dataset.py \
  --source-data-path "$SOURCE_DATA_PATH" \
  --debug-data-path "$DEBUG_DATA_PATH" \
  --train-tokens "$DEBUG_TRAIN_TOKENS" \
  --val-tokens "$DEBUG_VAL_TOKENS"

RUN_ID="${RUN_ID:-hf_verify_sp1024_cpu_debug}" \
DATA_PATH="${DATA_PATH:-$DEBUG_DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
ITERATIONS="${ITERATIONS:-100}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-5}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \
WARMUP_STEPS=0 \
python3 train_gpt_cpu.py
