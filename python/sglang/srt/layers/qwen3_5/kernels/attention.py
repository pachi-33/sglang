"""Packed-prefill and contiguous-cache GQA kernels for Qwen3.5."""

import math

import torch
import triton
import triton.language as tl

from .attention_slab import causal_gqa_slab

# Decode is deliberately a separate path from ``causal_gqa``.  The latter is
# a packed prefill kernel, while this path consumes the already materialised
# contiguous KV cache of one request.  Keeping the cache indexing here makes
# it impossible for the caller to accidentally turn a decode into a
# concatenate-and-prefill operation.
_DECODE_MAX_KV = 2048
_DECODE_BLOCK_N = 128
_DECODE_BLOCK_D = 16


@triton.jit
def _causal_gqa_decode_scores_kernel(
    q,
    k_cache,
    scores,
    kv_len,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Produce one FP32 score row for each of Qwen's 16 query heads."""
    head = tl.program_id(0)
    block = tl.program_id(1)
    offsets_n = block * BLOCK_N + tl.arange(0, BLOCK_N)
    kv_head = head // 8
    acc = tl.zeros((BLOCK_N,), tl.float32)
    # 256 is fixed by the Qwen3.5 Full Attention contract.  Loading K in
    # small tiles is SM70-safe and avoids relying on newer attention ops.
    for start_d in range(0, 256, BLOCK_K):
        offsets_d = start_d + tl.arange(0, BLOCK_K)
        q_tile = tl.load(q + head * 256 + offsets_d).to(tl.float32)
        k_tile = tl.load(
            k_cache
            + offsets_n[:, None] * (2 * 256)
            + kv_head * 256
            + offsets_d[None, :],
            mask=offsets_n[:, None] < kv_len,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(k_tile * q_tile[None, :], axis=1)
    tl.store(
        scores + head * _DECODE_MAX_KV + offsets_n,
        acc * SCALE,
        mask=offsets_n < kv_len,
    )


@triton.jit
def _causal_gqa_decode_pv_kernel(
    scores,
    v_cache,
    out,
    kv_len,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Softmax a score row and reduce its V values for one D tile."""
    head = tl.program_id(0)
    d_block = tl.program_id(1)
    offsets_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    max_score = -float("inf")
    # This fixed trip count is intentional: it keeps Triton 2.3's generated
    # SM70 code simple.  Every load is masked by the runtime valid length, so
    # cache capacity beyond ``kv_len`` is never read (and may contain NaNs).
    for start_n in range(0, _DECODE_MAX_KV, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        block_scores = tl.load(
            scores + head * _DECODE_MAX_KV + offsets_n,
            mask=offsets_n < kv_len,
            other=-float("inf"),
        )
        max_score = tl.maximum(max_score, tl.max(block_scores, axis=0))

    denom = 0.0
    acc = tl.zeros((BLOCK_D,), tl.float32)
    kv_head = head // 8
    for start_n in range(0, _DECODE_MAX_KV, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        block_scores = tl.load(
            scores + head * _DECODE_MAX_KV + offsets_n,
            mask=offsets_n < kv_len,
            other=-float("inf"),
        )
        probabilities = tl.exp(block_scores - max_score)
        probabilities = tl.where(offsets_n < kv_len, probabilities, 0.0)
        denom += tl.sum(probabilities, axis=0)
        values = tl.load(
            v_cache
            + offsets_n[:, None] * (2 * 256)
            + kv_head * 256
            + offsets_d[None, :],
            mask=offsets_n[:, None] < kv_len,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probabilities[:, None] * values, axis=0)
    tl.store(out + head * 256 + offsets_d, acc / denom)


@triton.jit
def _partial_neox_rope_kernel(
    x,
    positions,
    out,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    stride_t: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    position_stride: tl.constexpr,
    theta: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    value = tl.load(
        x + token * stride_t + head * stride_h + d * stride_d, mask=d < BLOCK_D
    )
    pair = d % 32
    mate = tl.where(d < 32, d + 32, d - 32)
    paired = tl.load(
        x + token * stride_t + head * stride_h + mate * stride_d, mask=d < 64, other=0.0
    ).to(tl.float32)
    position = tl.load(positions + token * position_stride).to(tl.float32)
    angle = position / tl.exp((pair.to(tl.float32) / 32.0) * tl.log(theta))
    cosine = tl.cos(angle)
    sine = tl.sin(angle)
    rotated = tl.where(
        d < 32,
        value.to(tl.float32) * cosine - paired * sine,
        paired * sine + value.to(tl.float32) * cosine,
    )
    tl.store(
        out + token * stride_t + head * stride_h + d * stride_d,
        tl.where(d < 64, rotated, value),
        mask=d < BLOCK_D,
    )


def causal_gqa(q, k, v, cu_seqlens, max_seqlen: int, softmax_scale=None, out=None):
    """Compute packed, ragged causal GQA without persistent KV state.

    This deliberately accepts the call-local ``cu_seqlens`` only; a cache or
    prefix length has no representation in the API.  This implementation
    supports at most 2048 packed tokens. The caller owns cu
    contents: it must start at zero, end at T, be nondecreasing, and each
    segment length must not exceed ``max_seqlen``.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("q must be [T,Hq,D] and k/v matching [T,Hkv,D]")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and cu_seqlens.is_cuda):
        raise ValueError("causal_gqa requires CUDA tensors")
    if not (q.device == k.device == v.device == cu_seqlens.device):
        raise ValueError("causal_gqa inputs must be on one CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != torch.float16:
        raise TypeError("causal_gqa currently supports FP16 only")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be a rank-1 int32 or int64 tensor")
    if not (q.is_contiguous() and k.is_contiguous() and cu_seqlens.is_contiguous()):
        raise ValueError("causal_gqa requires contiguous Q/K and cu_seqlens")
    if (
        v.stride(0) < v.shape[1] * v.shape[2]
        or v.stride(1) != v.shape[2]
        or v.stride(2) != 1
    ):
        raise ValueError("V must be a non-overlapping unit-inner-stride row view")
    tokens, hq, head_dim = q.shape
    if tokens > 2048:
        raise ValueError("causal_gqa supports at most 2048 packed tokens")
    if k.shape[0] != tokens or k.shape[2] != head_dim or hq % k.shape[1]:
        raise ValueError("invalid GQA head dimensions")
    if head_dim != 256 or hq != 16 or k.shape[1] != 2:
        raise ValueError("Qwen3.5 Full Attention requires Q16/KV2/D256")
    if isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int):
        raise TypeError("max_seqlen must be a Python int")
    if max_seqlen < 0 or max_seqlen > 2048:
        raise ValueError("max_seqlen must be in [0, 2048]")
    batch = cu_seqlens.numel() - 1
    if batch <= 0:
        raise ValueError("cu_seqlens must contain at least a start and end")
    if out is None:
        out = q.new_empty(q.shape)
    if (
        out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError("invalid attention output")
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if softmax_scale != head_dim**-0.5:
        raise ValueError("Qwen3.5 slab attention uses the fixed 1/sqrt(256) scale")
    if tokens == 0:
        if max_seqlen != 0:
            raise ValueError("empty q requires max_seqlen=0")
        return out
    if max_seqlen == 0:
        raise ValueError("nonempty q requires max_seqlen > 0")
    return causal_gqa_slab(q, k, v, cu_seqlens, max_seqlen, out)


def causal_gqa_decode(q_new, k_cache, v_cache, kv_len: int, scale=1.0 / 16.0):
    """Attend one Qwen3.5 query token over a contiguous request KV cache.

    ``k_cache`` and ``v_cache`` must already contain the new token at logical
    index ``kv_len - 1``.  Thus ``kv_len`` is the number of readable entries,
    not the old prefix length.  The function has no cache mutation and does
    not concatenate tensors; writing the new K/V is deliberately owned by the
    request-cache caller so that failures can be handled transactionally.
    """
    if q_new.shape != (1, 16, 256):
        raise ValueError("q_new must have shape [1, 16, 256]")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (2, 256):
        raise ValueError("k_cache must have shape [capacity, 2, 256]")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must match k_cache shape")
    if k_cache.shape[0] > _DECODE_MAX_KV:
        raise ValueError("causal_gqa_decode supports cache capacity at most 2048")
    if not (q_new.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("causal_gqa_decode requires CUDA tensors")
    if not (q_new.device == k_cache.device == v_cache.device):
        raise ValueError("causal_gqa_decode inputs must be on one CUDA device")
    if (
        q_new.dtype != torch.float16
        or k_cache.dtype != torch.float16
        or v_cache.dtype != torch.float16
    ):
        raise TypeError("causal_gqa_decode currently supports FP16 only")
    if not (
        q_new.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()
    ):
        raise ValueError("causal_gqa_decode requires contiguous Q and KV cache")
    if isinstance(kv_len, bool) or not isinstance(kv_len, int):
        raise TypeError("kv_len must be a Python int")
    if kv_len < 1 or kv_len > k_cache.shape[0]:
        raise ValueError("kv_len must be in [1, cache capacity]")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be a finite positive Python number")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if scale != 1.0 / 16.0:
        raise ValueError("Qwen3.5 decode attention scale is fixed at 1/16")

    # The score scratch is intentionally fixed-sized.  Both kernels mask all
    # reads with kv_len, while a fixed 2048 stride avoids a new Triton compile
    # variant for every prefix length.
    scores = torch.empty((16, _DECODE_MAX_KV), device=q_new.device, dtype=torch.float32)
    out = torch.empty_like(q_new)
    _causal_gqa_decode_scores_kernel[(16, triton.cdiv(kv_len, _DECODE_BLOCK_N))](
        q_new,
        k_cache,
        scores,
        kv_len,
        SCALE=scale,
        BLOCK_N=_DECODE_BLOCK_N,
        BLOCK_K=32,
        num_warps=4,
        num_stages=1,
    )
    _causal_gqa_decode_pv_kernel[(16, 256 // _DECODE_BLOCK_D)](
        scores,
        v_cache,
        out,
        kv_len,
        BLOCK_N=_DECODE_BLOCK_N,
        BLOCK_D=_DECODE_BLOCK_D,
        num_warps=4,
        num_stages=1,
    )
    return out


def partial_neox_rope(x, positions, theta: float = 10_000_000.0, out=None):
    """Apply Qwen3.5's 64-of-256 dimensional NeoX RoPE in Triton."""
    if x.ndim != 3 or x.shape[-1] != 256 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("RoPE expects contiguous CUDA [T,H,256]")
    if x.dtype != torch.float16:
        raise TypeError("RoPE currently supports FP16 only")
    if (
        positions.shape != (x.shape[0],)
        or not positions.is_cuda
        or not positions.is_contiguous()
    ):
        raise ValueError("positions must be contiguous CUDA [T]")
    if positions.device != x.device:
        raise ValueError("positions and x must be on one CUDA device")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must be int32 or int64")
    if not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    if out is None:
        out = x.new_empty(x.shape)
    if (
        out.shape != x.shape
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("invalid RoPE output")
    if x.numel() and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
        raise ValueError("RoPE output must not overlap x")
    if x.numel() == 0:
        return out
    _partial_neox_rope_kernel[(x.shape[0], x.shape[1])](
        x,
        positions,
        out,
        x.shape[0],
        x.shape[1],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        positions.stride(0),
        theta,
        BLOCK_D=256,
        num_warps=4,
    )
    return out
