#!/usr/bin/env bash
set -euo pipefail

# CPU variant of the baseline launcher. Keeps the same dataset, tokenizer, and
# cadence defaults where possible, but runs the single-process CPU debug script.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-hf_verify_sp1024_cpu}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \
python3 train_gpt_cpu.py
