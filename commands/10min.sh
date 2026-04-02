#!/usr/bin/env bash
set -euo pipefail

# Larger local MLX proxy run for a stronger signal than the 2-minute script.
# Still small enough for local iteration, but big enough to compare methods more meaningfully.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Each env var can be overridden at invocation time, e.g. `RUN_ID=test ./commands/10min.sh`.
# Use a larger batch/step budget while still avoiding periodic full validation.
# Use a larger validation subset than the 2-minute run, but still far smaller than full val.
RUN_ID="${RUN_ID:-mlx_cpu_proxy_10min}" \
SEED="${SEED:-314}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
ITERATIONS="${ITERATIONS:-250}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-16384}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}" \
MLX_MAX_MICROBATCH_TOKENS="${MLX_MAX_MICROBATCH_TOKENS:-4096}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}" \
WARMUP_STEPS="${WARMUP_STEPS:-0}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-131072}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}" \
python3 train_gpt_mlx.py
