"""Composition helpers for Qwen3.5 producers.

These functions allocate their output layouts directly and use only Triton
for CUDA data movement/elementwise work.  They deliberately keep projection
GEMMs separate: raw FP8 payloads must never be expanded into a dequantized
weight copy on Volta.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..weights import QuantActivation
from .quantization import _encode_e4m3fn


@triton.jit
def _full_qk_gate_kernel(
    qg,
    k,
    positions,
    q_weight,
    k_weight,
    q_out,
    k_out,
    gate_out,
    tokens: tl.constexpr,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SP: tl.constexpr,
    SWQ: tl.constexpr,
    SWK: tl.constexpr,
    theta: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 256)
    qbase = qg + token * SQ0 + head * 512 * SQ1
    kh = head % 2
    kval = k + token * SK0 + kh * 256 * SK1 + d * SK1
    qval = tl.load(qbase + d * SQ1).to(tl.float32)
    qss = tl.sum(qval * qval, axis=0)
    qscale = tl.load(q_weight + d * SWQ).to(tl.float32) + 1.0
    qval = (
        (qval * tl.math.rsqrt(qss / 256.0 + 1.0e-6) * qscale)
        .to(tl.float16)
        .to(tl.float32)
    )
    kval = tl.load(kval).to(tl.float32)
    kss = tl.sum(kval * kval, axis=0)
    kscale = tl.load(k_weight + d * SWK).to(tl.float32) + 1.0
    kval = (
        (kval * tl.math.rsqrt(kss / 256.0 + 1.0e-6) * kscale)
        .to(tl.float16)
        .to(tl.float32)
    )
    pair = d % 32
    mate = tl.where(d < 32, d + 32, d - 32)
    qmate = tl.load(qbase + mate * SQ1).to(tl.float32)
    kmate = tl.load(k + token * SK0 + kh * 256 * SK1 + mate * SK1).to(tl.float32)
    qmate = (
        (
            qmate
            * tl.math.rsqrt(qss / 256.0 + 1.0e-6)
            * (tl.load(q_weight + mate * SWQ).to(tl.float32) + 1.0)
        )
        .to(tl.float16)
        .to(tl.float32)
    )
    kmate = (
        (
            kmate
            * tl.math.rsqrt(kss / 256.0 + 1.0e-6)
            * (tl.load(k_weight + mate * SWK).to(tl.float32) + 1.0)
        )
        .to(tl.float16)
        .to(tl.float32)
    )
    pos = tl.load(positions + token * SP).to(tl.float32)
    angle = pos / tl.exp(pair.to(tl.float32) * tl.log(theta) / 32.0)
    c, s = tl.cos(angle), tl.sin(angle)
    qrot = tl.where(d < 32, qval * c - qmate * s, qmate * s + qval * c)
    krot = tl.where(d < 32, kval * c - kmate * s, kmate * s + kval * c)
    qresult = tl.where(d < 64, qrot, qval)
    kresult = tl.where(d < 64, krot, kval)
    tl.store(q_out + (token * 16 + head) * 256 + d, qresult.to(tl.float16))
    tl.store(k_out + (token * 2 + kh) * 256 + d, kresult.to(tl.float16), mask=head < 2)
    gate = tl.load(qbase + (256 + d) * SQ1)
    tl.store(gate_out + (token * 16 + head) * 256 + d, gate)


def full_qk_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    theta: float = 10_000_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Q/Gate, normalize Q/K and apply partial NeoX RoPE in one launch.

    ``q_gate`` is the physical Q-projection `[T, 16*512]`: every head holds
    Q[256] followed by its attention gate[256].  This is intentionally not a
    first/second-half split of the whole 8192-column projection.
    """
    if (
        q_gate.ndim != 2
        or q_gate.shape[1] != 8192
        or k.shape != (q_gate.shape[0], 512)
        or q_gate.dtype != torch.float16
        or k.dtype != torch.float16
    ):
        raise ValueError("expected FP16 q_gate[T,8192] and k[T,512]")
    if not all(t.is_cuda for t in (q_gate, k, positions, q_norm, k_norm)):
        raise ValueError("full_qk_rope_gate requires CUDA tensors")
    for source, name, width in ((q_gate, "q_gate", 8192), (k, "k", 512)):
        row_stride, inner_stride = source.stride()
        if row_stride < width or inner_stride != 1:
            raise ValueError(
                f"{name} must be a non-overlapping unit-inner-stride row view"
            )
    if not all(t.is_contiguous() for t in (positions, q_norm, k_norm)):
        raise ValueError("positions and Q/K norm weights must be contiguous")
    if any(t.device != q_gate.device for t in (k, positions, q_norm, k_norm)):
        raise ValueError("full_qk_rope_gate inputs must share one CUDA device")
    if positions.shape != (q_gate.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("positions must be CUDA integer [T]")
    if (
        q_norm.shape != (256,)
        or k_norm.shape != (256,)
        or q_norm.dtype != torch.float16
        or k_norm.dtype != torch.float16
    ):
        raise ValueError("Q/K norm weights must be [256]")
    t = q_gate.shape[0]
    q = torch.empty((t, 16, 256), dtype=torch.float16, device=q_gate.device)
    kout = torch.empty((t, 2, 256), dtype=torch.float16, device=q_gate.device)
    gate = torch.empty_like(q)
    if t:
        _full_qk_gate_kernel[(t, 16)](
            q_gate,
            k,
            positions,
            q_norm,
            k_norm,
            q,
            kout,
            gate,
            t,
            q_gate.stride(0),
            q_gate.stride(1),
            k.stride(0),
            k.stride(1),
            positions.stride(0),
            q_norm.stride(0),
            k_norm.stride(0),
            theta,
            num_warps=4,
            num_stages=1,
        )
    return q, kout, gate


@triton.jit
def _full_v_layout_kernel(source, out, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    value = tl.load(source + token * 512 + head * 256 + d)
    tl.store(out + (token * 2 + head) * 256 + d, value)


def split_full_v(v: torch.Tensor) -> torch.Tensor:
    """Write contiguous `[T,2,256]` V without a Torch view in the hot path."""
    if (
        v.ndim != 2
        or v.shape[1] != 512
        or v.dtype != torch.float16
        or not v.is_cuda
        or not v.is_contiguous()
    ):
        raise ValueError("v must be contiguous CUDA FP16 [T,512]")
    out = torch.empty((v.shape[0], 2, 256), dtype=v.dtype, device=v.device)
    if v.shape[0]:
        _full_v_layout_kernel[(v.shape[0], 2)](v, out, BLOCK=256, num_warps=4)
    return out


@triton.jit
def _gated_attention_a8_kernel(
    attn,
    gate,
    q,
    scale,
    boundary,
    groups: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPTURE_BOUNDARY: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    offset = token * 4096 + group * BLOCK + d
    value = tl.load(attn + offset).to(tl.float32)
    gate_value = tl.load(gate + offset).to(tl.float32)
    # Preserve the checkpoint's FP16 gate boundary before group quantization.
    half_value = (value / (1.0 + tl.exp(-gate_value))).to(tl.float16)
    if CAPTURE_BOUNDARY:
        tl.store(boundary + offset, half_value)
    # Frozen format contract: RN32 reciprocal multiply for local SF.  The
    # quotient below remains div_rn; the two operations are not interchangeable
    # at E4M3 decision boundaries.
    scale_value = tl.max(tl.abs(half_value.to(tl.float32)), axis=0) * (1.0 / 448.0)
    safe = tl.where(scale_value == 0.0, 1.0, scale_value)
    tl.store(
        q + offset, _encode_e4m3fn(tl.math.div_rn(half_value.to(tl.float32), safe))
    )
    tl.store(scale + token * groups + group, scale_value)


def gated_attention_fp8(
    attention: torch.Tensor, gate: torch.Tensor, *, capture_boundary: bool = False
):
    """Apply the FP16 sigmoid boundary and make the single A8 payload for O."""
    if (
        attention.shape != gate.shape
        or attention.ndim != 3
        or attention.shape[1:] != (16, 256)
    ):
        raise ValueError("attention and gate must be [T,16,256]")
    if (
        not all(
            x.is_cuda and x.is_contiguous() and x.dtype == torch.float16
            for x in (attention, gate)
        )
        or gate.device != attention.device
    ):
        raise ValueError("gated attention requires contiguous CUDA FP16 tensors")
    tokens = attention.shape[0]
    data = torch.empty((tokens, 4096), dtype=torch.uint8, device=attention.device)
    scale = torch.empty((tokens, 32), dtype=torch.float32, device=attention.device)
    boundary = (
        torch.empty((tokens, 4096), dtype=torch.float16, device=attention.device)
        if capture_boundary
        else data
    )
    if tokens:
        _gated_attention_a8_kernel[(tokens, 32)](
            attention,
            gate,
            data,
            scale,
            boundary,
            32,
            BLOCK=128,
            CAPTURE_BOUNDARY=capture_boundary,
            num_warps=4,
            num_stages=1,
        )
    payload = QuantActivation("fp8", data, (tokens, 4096), scale)
    return (payload, boundary) if capture_boundary else payload


@triton.jit
def _gdn_layout_kernel(
    source, q, k, v, src_stride: tl.constexpr, KIND: tl.constexpr, BLOCK: tl.constexpr
):
    token = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    if KIND == 0:
        offset = h * 128 + d
        target = q + (token * 16 + h) * 128 + d
    elif KIND == 1:
        offset = 2048 + h * 128 + d
        target = k + (token * 16 + h) * 128 + d
    else:
        offset = 4096 + h * 128 + d
        target = v + (token * 32 + h) * 128 + d
    value = tl.load(source + token * src_stride + offset, mask=d < 128)
    tl.store(target, value, mask=d < 128)


def split_gdn_qkv(
    conv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize the GDN Q/K/V head layouts from conv[Q,K,V] directly."""
    if (
        conv.ndim != 2
        or conv.shape[1] != 8192
        or conv.dtype != torch.float16
        or not conv.is_cuda
        or conv.stride() != (8192, 1)
    ):
        raise ValueError("conv must be canonical contiguous CUDA FP16 [T,8192]")
    t = conv.shape[0]
    q = torch.empty((t, 16, 128), dtype=conv.dtype, device=conv.device)
    k = torch.empty_like(q)
    v = torch.empty((t, 32, 128), dtype=conv.dtype, device=conv.device)
    if t:
        _gdn_layout_kernel[(t, 16)](
            conv, q, k, v, conv.stride(0), KIND=0, BLOCK=128, num_warps=4
        )
        _gdn_layout_kernel[(t, 16)](
            conv, q, k, v, conv.stride(0), KIND=1, BLOCK=128, num_warps=4
        )
        _gdn_layout_kernel[(t, 32)](
            conv, q, k, v, conv.stride(0), KIND=2, BLOCK=128, num_warps=4
        )
    return q, k, v


@triton.jit
def _gdn_norm_z_kernel(value, z, weight, out, heads: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    # The recurrent backend deliberately exposes FP32.  Qwen's next boundary
    # is FP16, so round before computing the RMS denominator.
    x16 = tl.load(value + (token * heads + head) * 128 + d).to(tl.float16)
    x = x16.to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / 128.0 + 1.0e-6)
    norm = x * inv * tl.load(weight + d).to(tl.float32)
    gate = tl.load(z + (token * heads + head) * 128 + d).to(tl.float32)
    tl.store(
        out + (token * heads + head) * 128 + d,
        (norm * gate / (1.0 + tl.exp(-gate))).to(tl.float16),
    )


@triton.jit
def _gdn_norm_z_a8_kernel(
    value,
    z,
    weight,
    q,
    scale,
    boundary,
    SZ: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPTURE_BOUNDARY: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    x = tl.load(value + (token * 32 + head) * 128 + d).to(tl.float16).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / 128.0 + 1.0e-6)
    norm = x * inv * tl.load(weight + d).to(tl.float32)
    gate = tl.load(z + token * SZ + head * 128 + d).to(tl.float32)
    result = (norm * gate / (1.0 + tl.exp(-gate))).to(tl.float16)
    if CAPTURE_BOUNDARY:
        tl.store(boundary + token * 4096 + head * 128 + d, result)
    sf = tl.max(tl.abs(result.to(tl.float32)), axis=0) * (1.0 / 448.0)
    safe = tl.where(sf == 0.0, 1.0, sf)
    offset = token * 4096 + head * 128 + d
    tl.store(q + offset, _encode_e4m3fn(tl.math.div_rn(result.to(tl.float32), safe)))
    tl.store(scale + token * 32 + head, sf)


def gated_gdn_fp8(
    value: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    capture_boundary: bool = False,
):
    """Apply the GDN FP32→FP16 norm/SiLU boundary and quantize its O input."""
    if value.ndim != 3 or value.shape[1:] != (32, 128) or value.dtype != torch.float32:
        raise ValueError("value must be CUDA FP32 [T,32,128]")
    if z.shape != (value.shape[0], 4096) or z.dtype != torch.float16:
        raise ValueError("z must be CUDA FP16 [T,4096]")
    if norm_weight.shape != (128,) or norm_weight.dtype != torch.float16:
        raise ValueError("norm_weight must be CUDA FP16 [128]")
    if (
        not all(x.is_cuda for x in (value, z, norm_weight))
        or not value.is_contiguous()
        or not norm_weight.is_contiguous()
        or z.device != value.device
        or norm_weight.device != value.device
    ):
        raise ValueError(
            "gated GDN producer requires CUDA tensors with contiguous value/norm_weight"
        )
    z_row_stride, z_inner_stride = z.stride()
    if z_row_stride < 4096 or z_inner_stride != 1:
        raise ValueError("z must be a non-overlapping unit-inner-stride row view")
    data = torch.empty((value.shape[0], 4096), dtype=torch.uint8, device=value.device)
    scale = torch.empty((value.shape[0], 32), dtype=torch.float32, device=value.device)
    boundary = (
        torch.empty((value.shape[0], 4096), dtype=torch.float16, device=value.device)
        if capture_boundary
        else data
    )
    if value.shape[0]:
        _gdn_norm_z_a8_kernel[(value.shape[0], 32)](
            value,
            z,
            norm_weight,
            data,
            scale,
            boundary,
            z.stride(0),
            BLOCK=128,
            CAPTURE_BOUNDARY=capture_boundary,
            num_warps=4,
            num_stages=1,
        )
    payload = QuantActivation("fp8", data, (value.shape[0], 4096), scale)
    return (payload, boundary) if capture_boundary else payload


@triton.jit
def _gather_hidden_kernel(
    hidden,
    indices,
    out,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    SH0: tl.constexpr,
    SH1: tl.constexpr,
    SI: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    index = tl.load(indices + row * SI)
    valid = (index >= 0) & (index < rows)
    value = tl.load(
        hidden + index * SH0 + d * SH1, mask=valid & (d < hidden_size), other=0.0
    )
    tl.store(out + row * SO0 + d * SO1, value, mask=d < hidden_size)


def gather_hidden(hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """GPU-only row gather used for arbitrary logit selections.

    Negative entries (used for an empty document in the default selection)
    produce a zero row.  Caller-provided positive indices are trusted metadata
    and receive the same bounds-safe behavior without a host synchronization.
    """
    if (
        hidden.ndim != 2
        or hidden.shape[1] != 2048
        or hidden.dtype != torch.float16
        or not hidden.is_cuda
        or not hidden.is_contiguous()
    ):
        raise ValueError("hidden must be contiguous CUDA FP16 [T,2048]")
    if (
        indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
        or not indices.is_cuda
        or not indices.is_contiguous()
        or indices.device != hidden.device
    ):
        raise ValueError("indices must be contiguous same-device CUDA integers")
    out = torch.empty((indices.numel(), 2048), dtype=hidden.dtype, device=hidden.device)
    if indices.numel():
        _gather_hidden_kernel[(indices.numel(),)](
            hidden,
            indices,
            out,
            hidden.shape[0],
            2048,
            hidden.stride(0),
            hidden.stride(1),
            indices.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK=2048,
            num_warps=4,
        )
    return out


@triton.jit
def _last_document_token_kernel(cu, indices, batch: tl.constexpr, CU: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(cu + row * CU)
    end = tl.load(cu + (row + 1) * CU)
    tl.store(indices + row, tl.where(end > start, end - 1, -1))


def default_last_token_indices(cu_seqlens: torch.Tensor, tokens: int) -> torch.Tensor:
    """Create one GPU index per document; an empty packed batch has zero rows."""
    if tokens == 0:
        return torch.empty((0,), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    batch = cu_seqlens.numel() - 1
    out = torch.empty((batch,), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    if batch:
        _last_document_token_kernel[(batch,)](
            cu_seqlens, out, batch, cu_seqlens.stride(0), num_warps=1
        )
    return out
