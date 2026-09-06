"""Stateless causal GQA kernel for Qwen3.5 Full-Attention layers."""

import math
import torch
import triton
import triton.language as tl
from .attention_slab import causal_gqa_slab


@triton.jit
def _partial_neox_rope_kernel(x, positions, out, tokens: tl.constexpr, heads: tl.constexpr,
                              stride_t: tl.constexpr, stride_h: tl.constexpr, stride_d: tl.constexpr,
                              position_stride: tl.constexpr, theta: tl.constexpr,
                              BLOCK_D: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    value = tl.load(x + token * stride_t + head * stride_h + d * stride_d, mask=d < BLOCK_D)
    pair = d % 32
    mate = tl.where(d < 32, d + 32, d - 32)
    paired = tl.load(x + token * stride_t + head * stride_h + mate * stride_d, mask=d < 64, other=0.0).to(tl.float32)
    position = tl.load(positions + token * position_stride).to(tl.float32)
    angle = position / tl.exp((pair.to(tl.float32) / 32.0) * tl.log(theta))
    cosine = tl.cos(angle)
    sine = tl.sin(angle)
    rotated = tl.where(d < 32, value.to(tl.float32) * cosine - paired * sine,
                       paired * sine + value.to(tl.float32) * cosine)
    tl.store(out + token * stride_t + head * stride_h + d * stride_d,
             tl.where(d < 64, rotated, value), mask=d < BLOCK_D)


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
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and cu_seqlens.is_contiguous()):
        raise ValueError("causal_gqa inputs must be contiguous")
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
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device or not out.is_contiguous():
        raise ValueError("invalid attention output")
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    if softmax_scale != head_dim ** -0.5:
        raise ValueError("Qwen3.5 slab attention uses the fixed 1/sqrt(256) scale")
    if tokens == 0:
        if max_seqlen != 0:
            raise ValueError("empty q requires max_seqlen=0")
        return out
    if max_seqlen == 0:
        raise ValueError("nonempty q requires max_seqlen > 0")
    return causal_gqa_slab(q, k, v, cu_seqlens, max_seqlen, out)


def partial_neox_rope(x, positions, theta: float = 10_000_000.0, out=None):
    """Apply Qwen3.5's 64-of-256 dimensional NeoX RoPE in Triton."""
    if x.ndim != 3 or x.shape[-1] != 256 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("RoPE expects contiguous CUDA [T,H,256]")
    if x.dtype != torch.float16:
        raise TypeError("RoPE currently supports FP16 only")
    if positions.shape != (x.shape[0],) or not positions.is_cuda or not positions.is_contiguous():
        raise ValueError("positions must be contiguous CUDA [T]")
    if positions.device != x.device:
        raise ValueError("positions and x must be on one CUDA device")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must be int32 or int64")
    if not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    if out is None:
        out = x.new_empty(x.shape)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError("invalid RoPE output")
    if x.numel() and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
        raise ValueError("RoPE output must not overlap x")
    if x.numel() == 0:
        return out
    _partial_neox_rope_kernel[(x.shape[0], x.shape[1])](
        x, positions, out, x.shape[0], x.shape[1], x.stride(0), x.stride(1), x.stride(2),
        positions.stride(0), theta, BLOCK_D=256, num_warps=4,
    )
    return out
