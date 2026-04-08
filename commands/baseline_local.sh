#!/usr/bin/env bash
set -euo pipefail

# Faster local proxy for the CUDA baseline command. Keeps the same PyTorch entrypoint
# and core dataset/tokenizer defaults, but trims the training/validation budget so
# short iteration runs are cheaper.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
RUN_ID="${RUN_ID:-hf_verify_sp1024_local}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
ITERATIONS="${ITERATIONS:-250}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py
