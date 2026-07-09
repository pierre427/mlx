"""Pure NumPy oracle and deterministic vectors for quantized SDPA.

The packed layout matches MLX affine quantization: little-endian codes inside
each uint32 word, followed by one scale and bias per last-axis group.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QSDPACase:
    name: str
    seed: int
    batch: int
    q_heads: int
    kv_heads: int
    q_len: int
    kv_len: int
    qk_dim: int
    value_dim: int
    bits: int
    group_size: int
    mask_kind: str
    mask_shape: tuple[int, ...] | None
    q_dtype: str = "float32"
    block_size: int = 17
    atol: float = 2e-5
    rtol: float = 2e-5


def affine_quantize(x: np.ndarray, group_size: int, bits: int):
    """Reference affine quantizer returning MLX-layout packed/scales/biases."""

    x = np.asarray(x, dtype=np.float32)
    _validate_quant_params(x.shape[-1], group_size, bits)
    grouped = x.reshape(*x.shape[:-1], -1, group_size)
    biases = grouped.min(axis=-1)
    maxima = grouped.max(axis=-1)
    levels = (1 << bits) - 1
    scales = (maxima - biases) / levels
    safe_scales = np.where(scales == 0, 1.0, scales)
    codes = np.rint((grouped - biases[..., None]) / safe_scales[..., None])
    codes = np.clip(codes, 0, levels).astype(np.uint32)
    scales = np.where(scales == 0, 0.0, scales).astype(np.float32)
    biases = biases.astype(np.float32)
    return pack_codes(codes.reshape(x.shape), bits), scales, biases


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack last-axis integer codes into little-endian uint32 words."""

    codes = np.asarray(codes, dtype=np.uint32)
    if bits not in (4, 8):
        raise ValueError("oracle packer supports bits in {4, 8}")
    per_word = 32 // bits
    if codes.shape[-1] % per_word:
        raise ValueError("code dimension must be divisible by codes per uint32")
    grouped = codes.reshape(*codes.shape[:-1], -1, per_word)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).reshape(
        *((1,) * (grouped.ndim - 1)), per_word
    )
    return np.bitwise_or.reduce(grouped << shifts, axis=-1, dtype=np.uint32)


def affine_dequantize(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int,
    bits: int,
) -> np.ndarray:
    """Dequantize MLX-layout packed affine values to float32."""

    packed = np.asarray(packed, dtype=np.uint32)
    scales = np.asarray(scales, dtype=np.float32)
    biases = np.asarray(biases, dtype=np.float32)
    if bits not in (4, 8):
        raise ValueError("oracle dequantizer supports bits in {4, 8}")
    per_word = 32 // bits
    output_dim = packed.shape[-1] * per_word
    _validate_quant_params(output_dim, group_size, bits)
    expected_param_shape = (*packed.shape[:-1], output_dim // group_size)
    if scales.shape != expected_param_shape or biases.shape != expected_param_shape:
        raise ValueError(
            "scales/biases shape must equal packed prefix plus D/group_size"
        )
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    codes = ((packed[..., None] >> shifts) & ((1 << bits) - 1)).reshape(
        *packed.shape[:-1], output_dim
    )
    grouped = codes.reshape(*packed.shape[:-1], -1, group_size)
    out = grouped * scales[..., None] + biases[..., None]
    return out.reshape(*packed.shape[:-1], output_dim).astype(np.float32)


def quantized_sdpa_online(
    q: np.ndarray,
    packed_k: np.ndarray,
    k_scales: np.ndarray,
    k_biases: np.ndarray,
    packed_v: np.ndarray,
    v_scales: np.ndarray,
    v_biases: np.ndarray,
    *,
    scale: float,
    group_size: int,
    bits: int,
    mask: Any = None,
    block_size: int = 64,
) -> np.ndarray:
    """Blockwise online-softmax qSDPA oracle; never materializes full scores."""

    q, k, v = validate_and_dequantize(
        q,
        packed_k,
        k_scales,
        k_biases,
        packed_v,
        v_scales,
        v_biases,
        group_size=group_size,
        bits=bits,
        mask=mask,
    )
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    batch, q_heads, q_len, _ = q.shape
    kv_heads, kv_len, value_dim = k.shape[1], k.shape[2], v.shape[-1]
    repeats = q_heads // kv_heads
    k = np.repeat(k, repeats, axis=1)
    v = np.repeat(v, repeats, axis=1)
    normalized_mask = normalize_mask(mask, batch, q_heads, q_len, kv_len)

    running_max = np.full((batch, q_heads, q_len), -np.inf, dtype=np.float32)
    running_sum = np.zeros_like(running_max)
    accumulator = np.zeros(
        (batch, q_heads, q_len, value_dim), dtype=np.float32
    )
    scaled_q = q.astype(np.float32) * np.float32(scale)
    for start in range(0, kv_len, block_size):
        stop = min(start + block_size, kv_len)
        scores = np.einsum(
            "bhld,bhsd->bhls", scaled_q, k[:, :, start:stop], optimize=True
        )
        block_mask = (
            None if normalized_mask is None else normalized_mask[..., start:stop]
        )
        scores = apply_mask(scores, block_mask)
        block_max = scores.max(axis=-1)
        next_max = np.maximum(running_max, block_max)
        finite_next = np.isfinite(next_max)
        max_delta = np.zeros_like(running_max)
        np.subtract(running_max, next_max, out=max_delta, where=finite_next)
        alpha = np.exp(max_delta)
        alpha = np.where(np.isfinite(running_max), alpha, 0.0)
        shifted = np.full_like(scores, -np.inf)
        np.subtract(
            scores, next_max[..., None], out=shifted, where=finite_next[..., None]
        )
        weights = np.exp(shifted).astype(np.float32)
        accumulator = accumulator * alpha[..., None] + np.einsum(
            "bhls,bhsv->bhlv", weights, v[:, :, start:stop], optimize=True
        )
        running_sum = running_sum * alpha + weights.sum(axis=-1)
        running_max = next_max
    return np.divide(
        accumulator,
        running_sum[..., None],
        out=np.zeros_like(accumulator),
        where=running_sum[..., None] != 0,
    )


def quantized_sdpa_dense(
    q: np.ndarray,
    packed_k: np.ndarray,
    k_scales: np.ndarray,
    k_biases: np.ndarray,
    packed_v: np.ndarray,
    v_scales: np.ndarray,
    v_biases: np.ndarray,
    *,
    scale: float,
    group_size: int,
    bits: int,
    mask: Any = None,
) -> np.ndarray:
    """Materialized dense reference used to verify the online recurrence."""

    q, k, v = validate_and_dequantize(
        q,
        packed_k,
        k_scales,
        k_biases,
        packed_v,
        v_scales,
        v_biases,
        group_size=group_size,
        bits=bits,
        mask=mask,
    )
    batch, q_heads, q_len, _ = q.shape
    kv_heads, kv_len = k.shape[1:3]
    repeats = q_heads // kv_heads
    k = np.repeat(k, repeats, axis=1)
    v = np.repeat(v, repeats, axis=1)
    scores = np.einsum(
        "bhld,bhsd->bhls", q.astype(np.float32) * scale, k, optimize=True
    )
    normalized_mask = normalize_mask(mask, batch, q_heads, q_len, kv_len)
    scores = apply_mask(scores, normalized_mask)
    maxima = scores.max(axis=-1, keepdims=True)
    weights = np.where(np.isfinite(maxima), np.exp(scores - maxima), 0.0)
    weights = np.divide(
        weights,
        weights.sum(axis=-1, keepdims=True),
        out=np.zeros_like(weights),
        where=weights.sum(axis=-1, keepdims=True) != 0,
    )
    return np.einsum("bhls,bhsv->bhlv", weights, v, optimize=True)


def validate_and_dequantize(
    q,
    packed_k,
    k_scales,
    k_biases,
    packed_v,
    v_scales,
    v_biases,
    *,
    group_size,
    bits,
    mask=None,
):
    q = np.asarray(q)
    packed_k = np.asarray(packed_k)
    packed_v = np.asarray(packed_v)
    if q.ndim != 4 or packed_k.ndim != 4 or packed_v.ndim != 4:
        raise ValueError("q and packed K/V must have rank 4 [B,H,L,D]")
    if packed_k.dtype != np.uint32 or packed_v.dtype != np.uint32:
        raise TypeError("packed K/V must have dtype uint32")
    if not np.issubdtype(q.dtype, np.floating):
        raise TypeError("q must have a floating dtype")
    batch, q_heads, q_len, qk_dim = q.shape
    if packed_k.shape[:3] != packed_v.shape[:3]:
        raise ValueError("packed K/V batch, head, and sequence dimensions must match")
    if packed_k.shape[0] != batch:
        raise ValueError("q and packed K/V batch dimensions must match")
    kv_heads, kv_len = packed_k.shape[1:3]
    if kv_heads < 1 or q_heads % kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    k = affine_dequantize(packed_k, k_scales, k_biases, group_size, bits)
    v = affine_dequantize(packed_v, v_scales, v_biases, group_size, bits)
    if k.shape[-1] != qk_dim:
        raise ValueError("dequantized K dimension must match q dimension")
    normalize_mask(mask, batch, q_heads, q_len, kv_len)
    return q, k, v


def normalize_mask(mask, batch: int, heads: int, q_len: int, kv_len: int):
    if mask is None:
        return None
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError("string mask must be 'causal'")
        q_indices = np.arange(q_len) + (kv_len - q_len)
        return q_indices[:, None] >= np.arange(kv_len)[None, :]
    mask = np.asarray(mask)
    try:
        return np.broadcast_to(mask, (batch, heads, q_len, kv_len))
    except ValueError as error:
        raise ValueError("mask is not broadcastable to [B,Hq,Lq,Lkv]") from error


def apply_mask(scores: np.ndarray, mask) -> np.ndarray:
    if mask is None:
        return scores
    if np.issubdtype(mask.dtype, np.bool_):
        return np.where(mask, scores, np.finfo(scores.dtype).min)
    return scores + mask


def correctness_cases() -> list[QSDPACase]:
    """Core 48-case matrix: bits × group × GQA ratio × mask kind."""

    cases = []
    for index, (bits, group_size, ratio, mask_kind) in enumerate(
        product((4, 8), (32, 64), (1, 2, 4), ("none", "bool", "additive", "causal"))
    ):
        q_len = 3 if mask_kind == "causal" else 1
        cases.append(
            QSDPACase(
                name=f"b{bits}_g{group_size}_r{ratio}_{mask_kind}",
                seed=1000 + index,
                batch=1,
                q_heads=2 * ratio,
                kv_heads=2,
                q_len=q_len,
                kv_len=67,
                qk_dim=128,
                value_dim=128,
                bits=bits,
                group_size=group_size,
                mask_kind=mask_kind,
                mask_shape=(1, 2 * ratio, q_len, 67)
                if mask_kind in ("bool", "additive")
                else None,
            )
        )
    return cases


def routing_boundary_cases() -> list[QSDPACase]:
    """Metadata for GPU dispatch sweeps; intentionally not run as CPU matmuls."""

    cases = []
    for bits, ratio, q_len, kv_len in product(
        (4, 8),
        (1, 2, 4),
        (1, 4, 8, 16, 31, 32, 33, 96, 127, 128, 129),
        (4096, 16384, 32768),
    ):
        cases.append(
            QSDPACase(
                name=f"route_b{bits}_r{ratio}_l{q_len}_s{kv_len}",
                seed=0,
                batch=1,
                q_heads=2 * ratio,
                kv_heads=2,
                q_len=q_len,
                kv_len=kv_len,
                qk_dim=128,
                value_dim=128,
                bits=bits,
                group_size=64,
                mask_kind="causal",
                mask_shape=None,
            )
        )
    return cases


def make_case_inputs(case: QSDPACase):
    rng = np.random.default_rng(case.seed)
    dtype = np.dtype(case.q_dtype)
    q = rng.normal(
        0, 0.25, (case.batch, case.q_heads, case.q_len, case.qk_dim)
    ).astype(dtype)
    k = rng.normal(
        0, 0.25, (case.batch, case.kv_heads, case.kv_len, case.qk_dim)
    ).astype(np.float32)
    v = rng.normal(
        0, 0.25, (case.batch, case.kv_heads, case.kv_len, case.value_dim)
    ).astype(np.float32)
    qk = affine_quantize(k, case.group_size, case.bits)
    qv = affine_quantize(v, case.group_size, case.bits)
    mask = None
    if case.mask_kind == "causal":
        mask = "causal"
    elif case.mask_kind == "bool":
        mask = rng.random(case.mask_shape) > 0.2
        mask[..., -1] = True
    elif case.mask_kind == "additive":
        mask = rng.uniform(-0.5, 0.0, case.mask_shape).astype(np.float32)
    elif case.mask_kind != "none":
        raise ValueError(f"unknown mask kind {case.mask_kind!r}")
    return q, qk, qv, mask


def _validate_quant_params(dim: int, group_size: int, bits: int) -> None:
    if bits not in (4, 8):
        raise ValueError("oracle supports bits in {4, 8}")
    if group_size not in (32, 64):
        raise ValueError("oracle supports group_size in {32, 64}")
    if dim % group_size:
        raise ValueError("last dimension must be divisible by group_size")
    if dim % (32 // bits):
        raise ValueError("last dimension is not packable into uint32 words")
