#!/usr/bin/env bash
set -euo pipefail

# Baseline CUDA training launch matching the standard multi-GPU PyTorch command.
# Intended for remote GPU machines; locally, override `NPROC_PER_NODE` if needed.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# NCCL_IB_DISABLE=1 disables InfiniBand transport. The remaining vars match the
# baseline dataset, tokenizer, cadence, and wallclock settings from the reference command.
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
RUN_ID="${RUN_ID:-hf_verify_sp1024_8gpu}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train_gpt.py
