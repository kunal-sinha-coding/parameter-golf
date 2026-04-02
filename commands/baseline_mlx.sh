#!/usr/bin/env bash
set -euo pipefail

# Closest single-process MLX equivalent to the PyTorch 8-GPU baseline launch.
# It preserves the same dataset/tokenizer/run settings and matches the global
# token budget using local gradient accumulation instead of distributed training.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Override any env var at invocation time if you want to change the batch shape or run id.
RUN_ID="${RUN_ID:-hf_verify_sp1024_8gpu}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}" \
python3 train_gpt_mlx.py
