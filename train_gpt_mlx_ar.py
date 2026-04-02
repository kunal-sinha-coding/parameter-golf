#!/usr/bin/env python3
"""
- Switches the MLX baseline to the AR-calibrated record-style stack: 11 layers, LeakyReLU(0.5)^2 MLP, partial RoPE, BigramHash, and XSA on all layers by default.
- Adds autoregressive self-generated calibration data collection so post-training quantization does not read train or validation tokens.
- Replaces the int8+zlib export path with an int6 + LZMA export path using Hessian-aware GPTQ-style row quantization when calibration Hessians are available.
- Adds size-targeted selective pruning of low-impact +/-1 quantized values so the compressed artifact can be pushed toward a requested byte budget.
- Marks all substantive changes with inline `EDIT:` comments so the diff is easy to audit.

Repo: https://github.com/kunal-sinha-coding/parameter-golf/blob/main/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md

Command:
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 TARGET_MB=15.9 SEED=314 python3 train_gpt_mlx_ar.py
"""
from __future__ import annotations

import glob
import json
import lzma
import math
import os
import pickle
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    # Optional local-debug cap for validation tokens. Keeps short runs short, but
    # makes val_loss / val_bpb only approximately comparable to full-val runs.
    val_max_tokens: int = int(os.environ.get("VAL_MAX_TOKENS", 0))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 2048)))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    # Force MLX to materialize the graph after every sub-batch, preventing lazy
    # graph buildup across accumulation steps. Keeps peak memory low on 16GB machines.
    # Disable on 32GB+ unified memory for better throughput (MLX_EAGER_EVAL=0).
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 4000))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(float(os.environ.get("MLP_MULT", 3.0)))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    bigram_vocab_size: int = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", 112))
    xsa_last_n: int = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims: int = int(os.environ.get("ROPE_DIMS", 16))
    bigram_scale_init: float = float(os.environ.get("BIGRAM_SCALE_INIT", 0.05))
    smear_enabled: bool = bool(int(os.environ.get("SMEAR_ENABLED", "1")))
    ve_enabled: bool = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim: int = int(os.environ.get("VE_DIM", 128))
    ve_layers: str = os.environ.get("VE_LAYERS", "9,10")
    calib_num_seqs: int = int(os.environ.get("CALIB_NUM_SEQS", 64))
    calib_batch_size: int = int(os.environ.get("CALIB_BATCH_SIZE", 8))
    calib_temperature: float = float(os.environ.get("CALIB_TEMPERATURE", 0.8))
    gptq_block_size: int = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
    target_mb: float = float(os.environ.get("TARGET_MB", 15.9))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled: bool = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every: int = int(os.environ.get("SWA_EVERY", 50))
    swa_start_lr_frac: float = float(os.environ.get("SWA_START_LR_FRAC", 0.2))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.035))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,bigram.scale,smear.gate,ve_layer_scales,ve_shared.scale",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class BigramHashEmbedding(nn.Module):
    # EDIT: Hash previous/current token pairs into a compact learned table and project it into model space.
    def __init__(self, vocab_size: int, bigram_vocab_size: int, bigram_dim: int, dim: int, scale_init: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, dim) if bigram_dim != dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(scale_init, dtype=mx.float32)

    def __call__(self, input_ids: mx.array) -> mx.array:
        t = input_ids.astype(mx.int32)
        mod = int(self.bigram_vocab_size) - 1
        first = mx.full((input_ids.shape[0], 1), mod, dtype=mx.int32)
        rest = ((36313 * t[:, 1:]) ^ (27191 * t[:, :-1])) % mod
        bigram_ids = mx.concatenate([first, rest.astype(mx.int32)], axis=1)
        x = self.embed(bigram_ids.astype(mx.int32)).astype(COMPUTE_DTYPE)
        if self.proj is not None:
            x = self.proj(x)
        return self.scale.astype(x.dtype) * x


class SmearGate(nn.Module):
    # EDIT: Blend each token with its previous token using a learned per-channel gate.
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        g = nn.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return (1.0 - g) * x + g * x_prev


class ValueEmbedding(nn.Module):
    # EDIT: Reinject token identity into attention values on a chosen subset of layers.
    def __init__(self, vocab_size: int, ve_dim: int, kv_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        self.embed.weight = (mx.random.normal(self.embed.weight.shape, dtype=mx.float32) * 0.01).astype(COMPUTE_DTYPE)
        self.proj = CastedLinear(ve_dim, kv_dim) if ve_dim != kv_dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(0.1, dtype=mx.float32)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed(input_ids).astype(COMPUTE_DTYPE)
        if self.proj is not None:
            x = self.proj(x)
        return self.scale.astype(x.dtype) * x


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - causal masked SDPA
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int,
        use_xsa: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if rope_dims < 0 or rope_dims > self.head_dim or rope_dims % 2 != 0:
            raise ValueError(f"rope_dims must be an even value in [0, head_dim], got {rope_dims}")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope_dims = rope_dims
        self.use_xsa = use_xsa
        self.rope = nn.RoPE(self.rope_dims, traditional=False, base=rope_base) if self.rope_dims > 0 else None
        self.scale = self.head_dim ** -0.5

    def apply_partial_rope(self, x: mx.array) -> mx.array:
        # EDIT: Only the leading rope_dims channels get rotated; the rest pass through untouched.
        if self.rope is None or self.rope_dims == 0:
            return x
        if self.rope_dims == self.head_dim:
            return self.rope(x)
        x_rope = self.rope(x[..., : self.rope_dims])
        return mx.concatenate([x_rope, x[..., self.rope_dims :]], axis=-1)

    def apply_xsa(self, y: mx.array, v: mx.array) -> mx.array:
        # EDIT: GQA-aware XSA removes the output component aligned with each token's own value vector.
        if not self.use_xsa:
            return y
        group = self.num_heads // self.num_kv_heads
        y_grouped = y.reshape(y.shape[0], self.num_kv_heads, group, y.shape[2], y.shape[3])
        v_grouped = v[:, :, None, :, :]
        numer = mx.sum(y_grouped * v_grouped, axis=-1, keepdims=True)
        denom = mx.sum(v_grouped * v_grouped, axis=-1, keepdims=True) + 1e-6
        y_grouped = y_grouped - (numer / denom) * v_grouped
        return y_grouped.reshape(y.shape)

    def __call__(
        self,
        x: mx.array,
        v_embed: mx.array | None = None,
        collector: dict[str, list[np.ndarray]] | None = None,
        prefix: str = "",
    ) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed.astype(v.dtype)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        if collector is not None:
            x_np = np.ascontiguousarray(np.array(x.astype(mx.float32)))
            collector.setdefault(f"{prefix}.c_q.weight", []).append(x_np.reshape(-1, x_np.shape[-1]))
            collector.setdefault(f"{prefix}.c_k.weight", []).append(x_np.reshape(-1, x_np.shape[-1]))
            collector.setdefault(f"{prefix}.c_v.weight", []).append(x_np.reshape(-1, x_np.shape[-1]))

        q = self.apply_partial_rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.apply_partial_rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = self.apply_xsa(y, v)
        if collector is not None:
            y_np = np.ascontiguousarray(np.array(y.transpose(0, 2, 1, 3).reshape(-1, dim).astype(mx.float32)))
            collector.setdefault(f"{prefix}.proj.weight", []).append(y_np)
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array, collector: dict[str, list[np.ndarray]] | None = None, prefix: str = "") -> mx.array:
        # EDIT: The record stack uses LeakyReLU(0.5)^2 instead of ReLU^2.
        if collector is not None:
            x_np = np.ascontiguousarray(np.array(x.astype(mx.float32)))
            collector.setdefault(f"{prefix}.fc.weight", []).append(x_np.reshape(-1, x_np.shape[-1]))
        h = nn.leaky_relu(self.fc(x), negative_slope=0.5)
        h = h * h
        if collector is not None:
            h_np = np.ascontiguousarray(np.array(h.astype(mx.float32)))
            collector.setdefault(f"{prefix}.proj.weight", []).append(h_np.reshape(-1, h_np.shape[-1]))
        return self.proj(h)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int,
        use_xsa: bool,
        layer_index: int,
        smear_enabled: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims, use_xsa)
        self.mlp = MLP(dim, mlp_mult)
        self.smear = SmearGate(dim) if smear_enabled else None
        ln_scale = 1.0 / math.sqrt(layer_index + 1.0)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32) * ln_scale
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32) * ln_scale
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(
        self,
        x: mx.array,
        x0: mx.array,
        v_embed: mx.array | None = None,
        collector: dict[str, list[np.ndarray]] | None = None,
        prefix: str = "",
    ) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.smear is not None:
            x = self.smear(x)
        attn_out = self.attn(self.attn_norm(x), v_embed=v_embed, collector=collector, prefix=f"{prefix}.attn")
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x), collector=collector, prefix=f"{prefix}.mlp")
        return x


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - tied embeddings for the LM head (the baseline default setup)
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, bigram_vocab_size: int, bigram_dim: int, xsa_last_n: int, rope_dims: int,
                 bigram_scale_init: float, smear_enabled: bool, ve_enabled: bool, ve_dim: int, ve_layers: str):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, dim)
        # EDIT: Add a hashed bigram side-channel to the tied embedding pathway.
        self.bigram = BigramHashEmbedding(vocab_size, bigram_vocab_size, bigram_dim, dim, bigram_scale_init)
        self.kv_dim = self.num_kv_heads = num_kv_heads
        self._ve_target_dim = num_kv_heads * (dim // num_heads)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        self.ve_layer_to_scale_idx = {layer_idx: i for i, layer_idx in enumerate(self.ve_layer_indices)}
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, self._ve_target_dim)
            self.ve_layer_scales = [
                mx.ones((1,), dtype=mx.float32)
                for _ in self.ve_layer_indices
            ]
        else:
            self.ve_shared = None
            self.ve_layer_scales = []
        self.blocks = [
            Block(
                dim,
                num_heads,
                num_kv_heads,
                mlp_mult,
                rope_base,
                qk_gain_init,
                rope_dims,
                i >= max(num_layers - xsa_last_n, 0),
                i,
                smear_enabled,
            )
            for i in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm((self.tok_emb(input_ids) + self.bigram(input_ids)).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []
        ve_base = self.ve_shared(input_ids) if self.ve_shared is not None else None

        for i in range(self.num_encoder_layers):
            ve = None
            if ve_base is not None and i in self.ve_layer_to_scale_idx:
                ve = ve_base * self.ve_layer_scales[self.ve_layer_to_scale_idx[i]].astype(ve_base.dtype)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            # Odd layer counts have one more decoder block than encoder block. The baseline only
            # applies a skip connection when one exists, then runs the remaining decoder block(s)
            # without an added skip.
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            block_idx = self.num_encoder_layers + i
            ve = None
            if ve_base is not None and block_idx in self.ve_layer_to_scale_idx:
                ve = ve_base * self.ve_layer_scales[self.ve_layer_to_scale_idx[block_idx]].astype(ve_base.dtype)
            x = self.blocks[block_idx](x, x0, v_embed=ve)
        return self.final_norm(x)

    def forward_with_collector(self, input_ids: mx.array, collector: dict[str, list[np.ndarray]] | None = None) -> mx.array:
        # EDIT: This uncompiled path mirrors the forward pass but records linear-layer inputs for AR GPTQ calibration.
        x = rms_norm((self.tok_emb(input_ids) + self.bigram(input_ids)).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []
        ve_base = self.ve_shared(input_ids) if self.ve_shared is not None else None
        for i in range(self.num_encoder_layers):
            ve = None
            if ve_base is not None and i in self.ve_layer_to_scale_idx:
                ve = ve_base * self.ve_layer_scales[self.ve_layer_to_scale_idx[i]].astype(ve_base.dtype)
            x = self.blocks[i](x, x0, v_embed=ve, collector=collector, prefix=f"blocks.{i}")
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            block_idx = self.num_encoder_layers + i
            ve = None
            if ve_base is not None and block_idx in self.ve_layer_to_scale_idx:
                ve = ve_base * self.ve_layer_scales[self.ve_layer_to_scale_idx[block_idx]].astype(ve_base.dtype)
            x = self.blocks[block_idx](x, x0, v_embed=ve, collector=collector, prefix=f"blocks.{block_idx}")
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_keys = ["tok_emb.weight", "bigram.embed.weight"]
        if model.bigram.proj is not None:
            self.embed_keys.append("bigram.proj.weight")
        if model.ve_shared is not None:
            self.embed_keys.append("ve_shared.embed.weight")
            if model.ve_shared.proj is not None:
                self.embed_keys.append("ve_shared.proj.weight")
        self.matrix_keys = [
            k
            for k, p in params.items()
            if (
                (k.startswith("blocks.") or k == "bigram.proj.weight" or k == "ve_shared.proj.weight")
                and p.ndim == 2
                and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            )
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if (
                k == "skip_weights"
                or k == "bigram.scale"
                or k == "smear.gate"
                or k == "ve_shared.scale"
                or k.startswith("ve_layer_scales.")
                or (k.startswith("blocks.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
            )
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {k: grads[k] for k in self.embed_keys},
                {k: params[k] for k in self.embed_keys},
            )
        )

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (AR GPTQ INT6 + LZMA)
# ==============================================================================
# - 2D tensors use int6 row-wise quantization
# - when AR calibration Hessians are available, use a GPTQ-style preconditioned solve
# - small/control tensors stay in float form
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT6_MAX = 31
INT6_KEEP_FLOAT_MAX_NUMEL = 65_536
INT6_KEEP_FLOAT_STORE_DTYPE = np.float16
INT6_PER_ROW_SCALE_DTYPE = np.float16
INT6_CLIP_PERCENTILE = 99.99984
INT6_CLIP_Q = INT6_CLIP_PERCENTILE / 100.0


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT6_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_row_int6(row: np.ndarray) -> tuple[np.ndarray, np.float32]:
    clip_abs = float(np.quantile(np.abs(row), INT6_CLIP_Q)) if row.size else 0.0
    scale = np.float32(max(clip_abs / INT6_MAX, 1.0 / INT6_MAX))
    q = np.clip(np.round(np.clip(row, -clip_abs, clip_abs) / scale), -INT6_MAX, INT6_MAX).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def gptq_quantize_matrix(weight: np.ndarray, hessian: np.ndarray | None, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    # EDIT: Apply a compact Full-Hessian GPTQ-style solve row-by-row when calibration Hessians exist.
    w = np.ascontiguousarray(weight.astype(np.float32, copy=False))
    q = np.empty_like(w, dtype=np.int8)
    scales = np.empty((w.shape[0],), dtype=np.float32)
    if hessian is None or hessian.shape[0] != w.shape[1]:
        for row_idx in range(w.shape[0]):
            q[row_idx], scales[row_idx] = quantize_row_int6(w[row_idx])
        return q, scales

    h = np.array(hessian, dtype=np.float64, copy=True)
    h.flat[:: h.shape[0] + 1] += 1e-4
    inv = np.linalg.inv(h)
    for row_idx in range(w.shape[0]):
        row = w[row_idx].astype(np.float64, copy=True)
        q_row = np.empty((w.shape[1],), dtype=np.int8)
        _, scale = quantize_row_int6(row.astype(np.float32, copy=False))
        scales[row_idx] = scale
        for block_start in range(0, w.shape[1], block_size):
            block_end = min(block_start + block_size, w.shape[1])
            for col in range(block_start, block_end):
                q_val = np.clip(np.round(row[col] / scale), -INT6_MAX, INT6_MAX)
                q_row[col] = np.int8(q_val)
                err = row[col] - float(q_val) * float(scale)
                row -= err * inv[:, col] / max(inv[col, col], 1e-8)
        q[row_idx] = q_row
    return np.ascontiguousarray(q), np.ascontiguousarray(scales.astype(INT6_PER_ROW_SCALE_DTYPE, copy=False))


def quantize_vector_int6(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clip_abs = float(np.quantile(np.abs(arr).reshape(-1), INT6_CLIP_Q)) if arr.size else 0.0
    scale = np.array(clip_abs / INT6_MAX if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(arr, -clip_abs, clip_abs) / scale), -INT6_MAX, INT6_MAX).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int6(
    flat_state: dict[str, mx.array],
    hessians: dict[str, np.ndarray] | None,
    block_size: int,
) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int6_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int6_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        if int(arr.size) <= INT6_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        f32 = _np_float32(arr)
        if f32.ndim == 2:
            q, s = gptq_quantize_matrix(f32, None if hessians is None else hessians.get(name), block_size)
        else:
            q, s = quantize_vector_int6(f32)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int6_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int6_gptq_ar_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def selective_prune_quantized_int6(
    quant_obj: dict[str, object],
    target_total_bytes: int,
    code_bytes: int,
) -> dict[str, object]:
    # EDIT: Drop the least expensive +/-1 entries first, mirroring the record's size-targeted pruning pass.
    current_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    current_total = len(lzma.compress(current_raw, preset=9)) + code_bytes
    if current_total <= target_total_bytes:
        return quant_obj
    work = pickle.loads(pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL))
    candidates: list[tuple[float, str, int]] = []
    for name, q in work["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        scale = np.asarray(work["scales"][name], dtype=np.float32)
        if scale.ndim == 0:
            errors = np.full(q_np.shape, float(scale * scale), dtype=np.float32)
        else:
            errors = np.broadcast_to(scale[:, None] * scale[:, None], q_np.shape)
        mask = np.abs(q_np) == 1
        if np.any(mask):
            flat_idx = np.flatnonzero(mask.reshape(-1))
            flat_err = errors.reshape(-1)[flat_idx]
            candidates.extend((float(err), name, int(idx)) for err, idx in zip(flat_err.tolist(), flat_idx.tolist(), strict=True))
    candidates.sort(key=lambda x: x[0])
    for _, name, flat_idx in candidates:
        np.asarray(work["quantized"][name]).reshape(-1)[flat_idx] = 0
        trial_raw = pickle.dumps(work, protocol=pickle.HIGHEST_PROTOCOL)
        trial_total = len(lzma.compress(trial_raw, preset=9)) + code_bytes
        if trial_total <= target_total_bytes:
            return work
    return work


def dequantize_state_dict_int6(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8).astype(np.float32)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def sample_next_token(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> np.ndarray:
    if temperature <= 0:
        return np.argmax(logits, axis=-1).astype(np.int32, copy=False)
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.array([rng.choice(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int32)


def generate_autoregressive_calib_tokens(model: GPT, args: Hyperparameters) -> np.ndarray:
    # EDIT: Generate calibration sequences from the model itself so quantization remains self-contained.
    rng = np.random.default_rng(args.seed)
    tokens = np.zeros((args.calib_num_seqs, args.train_seq_len), dtype=np.int32)
    tokens[:, 0] = rng.integers(0, args.vocab_size, size=(args.calib_num_seqs,), dtype=np.int32)
    for start in range(0, args.calib_num_seqs, args.calib_batch_size):
        end = min(start + args.calib_batch_size, args.calib_num_seqs)
        batch = tokens[start:end]
        for pos in range(1, args.train_seq_len):
            hidden = model(mx.array(batch[:, :pos], dtype=mx.int32))
            logits = np.array((hidden[:, -1, :] @ model.tok_emb.weight.astype(hidden.dtype).T).astype(mx.float32))
            logits = args.logit_softcap * np.tanh(logits / args.logit_softcap)
            batch[:, pos] = sample_next_token(logits, args.calib_temperature, rng)
        tokens[start:end] = batch
    return tokens


def collect_hessians_from_tokens(model: GPT, token_batches: np.ndarray) -> dict[str, np.ndarray]:
    # EDIT: Replay AR-generated sequences through an instrumented forward pass and accumulate X^T X per linear layer.
    collector: dict[str, list[np.ndarray]] = {}
    for batch in token_batches:
        model.forward_with_collector(mx.array(batch[None, :], dtype=mx.int32), collector=collector)
    hessians: dict[str, np.ndarray] = {}
    for name, chunks in collector.items():
        x = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
        hessians[name] = np.ascontiguousarray((x.T @ x) / max(x.shape[0], 1))
    return hessians


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest
    # lets the training script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int, max_tokens: int = 0) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    if max_tokens > 0:
        limited = min(tokens.size - 1, max_tokens)
        usable_limited = (limited // seq_len) * seq_len
        if usable_limited <= 0:
            raise ValueError(f"VAL_MAX_TOKENS={max_tokens} is too small for TRAIN_SEQ_LEN={seq_len}")
        tokens = tokens[: usable_limited + 1]
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)  # materialize each chunk to cap peak memory
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb

# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def snapshot_flat_state_np(model: nn.Module) -> dict[str, np.ndarray]:
    flat = dict(tree_flatten(model.state))
    snap: dict[str, np.ndarray] = {}
    for name, arr in flat.items():
        if mx.issubdtype(arr.dtype, mx.floating):
            snap[name] = np.array(arr.astype(mx.float32), copy=True)
        else:
            snap[name] = np.array(arr, copy=True)
    return snap


def load_flat_state_np(model: nn.Module, flat_np: dict[str, np.ndarray]) -> None:
    current = dict(tree_flatten(model.state))
    restored = []
    for name, arr in current.items():
        restored.append((name, mx.array(flat_np[name], dtype=arr.dtype)))
    model.update(tree_unflatten(restored))


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_mlx_ar.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, max_tokens=args.val_max_tokens)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # ==============================================================================
    # MODEL + OPTIMIZER SETUP
    # ==============================================================================
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        bigram_scale_init=args.bigram_scale_init,
        smear_enabled=args.smear_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    )
    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
    # ==============================================================================
    # The crucial MLX detail is capture scope: this model contains non-trainable arrays too (for example
    # inside RoPE modules), so compiling only against trainable parameters throws "uncaptured inputs".
    # Compiling the model-bound functions and capturing the full model state fixes that while still
    # returning gradients only for trainable parameters via nn.value_and_grad(...).
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if args.val_max_tokens > 0:
        log(f"WARNING: validation_subset_enabled val_max_tokens:{args.val_max_tokens}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    # EDIT: Surface the record-style architecture knobs directly in the log header.
    log(
        f"bigram_vocab_size:{args.bigram_vocab_size} bigram_dim:{args.bigram_dim} "
        f"xsa_last_n:{args.xsa_last_n} rope_dims:{args.rope_dims} "
        f"smear_enabled:{args.smear_enabled} ve_enabled:{args.ve_enabled} ve_layers:{args.ve_layers} "
        f"calib_num_seqs:{args.calib_num_seqs} calib_temperature:{args.calib_temperature}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    if args.warmup_steps > 0:
        # Warmup should only prime MLX compile/allocation paths. Updating parameters here forces us
        # to snapshot and restore model/optimizer state, which is expensive on unified-memory Macs.
        # Instead we run the real train shapes, force the loss/grads to materialize, and then reset
        # the loader so measured training still starts from the true init and token window.
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0
    ema_state = snapshot_flat_state_np(model)
    swa_state: dict[str, np.ndarray] | None = None
    swa_count = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
            )
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)  # materialize each microbatch to cap peak memory

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()
        current_flat_np = snapshot_flat_state_np(model)
        for name, arr in current_flat_np.items():
            if np.issubdtype(arr.dtype, np.floating):
                ema_state[name] *= args.ema_decay
                ema_state[name] += arr * (1.0 - args.ema_decay)
            else:
                ema_state[name] = arr
        if args.swa_enabled and lr_mul < args.swa_start_lr_frac and (step + 1) % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: arr.astype(np.float32, copy=True) if np.issubdtype(arr.dtype, np.floating) else arr.copy() for name, arr in current_flat_np.items()}
                swa_count = 1
                log(f"swa:start step:{step + 1}")
            else:
                for name, arr in current_flat_np.items():
                    if np.issubdtype(arr.dtype, np.floating):
                        swa_state[name] += arr.astype(np.float32, copy=False)
                    else:
                        swa_state[name] = arr
                swa_count += 1

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    log("ema:applying averaged weights")
    final_avg_state = {name: arr.copy() for name, arr in ema_state.items()}
    if args.swa_enabled and swa_state is not None and swa_count > 0:
        log(f"swa:blending {swa_count} late checkpoints with EMA")
        for name, arr in final_avg_state.items():
            if np.issubdtype(arr.dtype, np.floating):
                final_avg_state[name] = 0.5 * arr + 0.5 * (swa_state[name] / float(swa_count))
    load_flat_state_np(model, final_avg_state)
    mx.synchronize()

    # ==============================================================================
    # FINAL SERIALIZATION + AR-GPTQ ROUNDTRIP EVAL
    # ==============================================================================
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    # EDIT: Build GPTQ calibration statistics from self-generated tokens rather than any dataset split.
    calib_tokens = generate_autoregressive_calib_tokens(model, args)
    log(
        f"gptq:generated_ar_calibration num_seqs:{args.calib_num_seqs} "
        f"seq_len:{args.train_seq_len} temperature:{args.calib_temperature}"
    )
    hessians = collect_hessians_from_tokens(model, calib_tokens)
    log(f"gptq:collected_hessians tensors:{len(hessians)}")

    quant_obj, quant_stats = quantize_state_dict_int6(flat_state, hessians=hessians, block_size=args.gptq_block_size)
    quant_obj = selective_prune_quantized_int6(
        quant_obj,
        target_total_bytes=int(args.target_mb * 1024 * 1024),
        code_bytes=len(code.encode("utf-8")),
    )
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = lzma.compress(quant_raw, preset=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int6.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int6_payload_bytes"], 1)
    log(
        f"serialized_model_int6_lzma:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int6_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x "
        f"total_with_code:{quant_file_bytes + len(code.encode('utf-8'))})"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int6(pickle.loads(lzma.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int6_lzma_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int6_lzma_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
