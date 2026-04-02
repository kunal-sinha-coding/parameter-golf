#!/usr/bin/env bash
set -euo pipefail

# Fairer local MLX-AR proxy than the 2-minute script.
# Uses the AR stack, but keeps sequence length at 1024 locally so step count and
# effective supervision are closer to the baseline proxy on small CPU runs.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Each env var can be overridden at invocation time, e.g. `RUN_ID=test ./commands/10min_ar.sh`.
# This is still a proxy, not a faithful reproduction of the 8xH100 record run.
RUN_ID="${RUN_ID:-mlx_ar_cpu_proxy_10min}" \
SEED="${SEED:-314}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}" \
BIGRAM_DIM="${BIGRAM_DIM:-112}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
TARGET_MB="${TARGET_MB:-15.9}" \
ITERATIONS="${ITERATIONS:-250}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-16384}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}" \
MLX_MAX_MICROBATCH_TOKENS="${MLX_MAX_MICROBATCH_TOKENS:-4096}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}" \
WARMUP_STEPS="${WARMUP_STEPS:-0}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-131072}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}" \
python3 train_gpt_mlx_ar.py
