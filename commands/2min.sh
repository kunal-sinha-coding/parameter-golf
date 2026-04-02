#!/usr/bin/env bash
set -euo pipefail

# Quick local MLX debug run sized for roughly a couple of minutes on a CPU.
# Uses a tiny token budget and no periodic validation so you can catch obvious
# regressions or instability before paying for a larger run.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Each env var can be overridden at invocation time, e.g. `SEED=42 ./commands/2min.sh`.
# Keep the run short and the effective batch small enough for local debugging.
# Skip mid-run validation sweeps; the script still reports final val metrics once.
# Cap final validation to a small subset so this stays genuinely quick on CPU.
RUN_ID="${RUN_ID:-mlx_cpu_debug_2min}" \
SEED="${SEED:-314}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
ITERATIONS="${ITERATIONS:-150}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}" \
MLX_MAX_MICROBATCH_TOKENS="${MLX_MAX_MICROBATCH_TOKENS:-4096}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}" \
WARMUP_STEPS="${WARMUP_STEPS:-0}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-32768}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}" \
python3 train_gpt_mlx.py
