"""Raw-layout FP8/NVFP4 helpers for the V100 Qwen3.5 compatibility path.

Checkpoint FP8 and FP4 bytes are intentionally retained.  These utilities only
create per-call activation payloads; they never materialize a dequantized copy
of a model weight.
"""

from __future__ import annotations

import torch
import triton

from .kernels.quantization import (
    decode_e2m1_kernel,
    decode_e4m3fn_kernel,
    encode_e2m1_kernel,
    encode_e4m3fn_kernel,
    fp8_block128_gemm_kernel,
    quantize_fp8_group128_kernel,
    quantize_nvfp4_group16_kernel,
)
from .weights import QuantActivation, Weight

E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0**-9
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _as_u8(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.uint8 else x.view(torch.uint8)


def decode_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    """Decode E4M3FN bytes to FP32, preserving the two reserved NaN codes."""
    if x.is_cuda:
        raw = _as_u8(x)
        if not raw.is_contiguous():
            raise ValueError("CUDA E4M3FN decode requires contiguous storage")
        out = torch.empty(raw.shape, device=raw.device, dtype=torch.float32)
        decode_e4m3fn_kernel[(triton.cdiv(raw.numel(), 256),)](
            raw, out, numel=raw.numel(), BLOCK=256
        )
        return out
    b = _as_u8(x).to(torch.int32)
    sign = torch.where((b & 0x80) != 0, -1.0, 1.0)
    exponent = (b >> 3) & 0xF
    mantissa = b & 0x7
    normal = torch.exp2(exponent.to(torch.float32) - 7) * (
        1.0 + mantissa.to(torch.float32) / 8.0
    )
    subnormal = mantissa.to(torch.float32) * E4M3_MIN_SUBNORMAL
    value = sign * torch.where(exponent == 0, subnormal, normal)
    return torch.where(
        (exponent == 15) & (mantissa == 7), torch.full_like(value, float("nan")), value
    )


def encode_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    """Software E4M3FN RNE/saturating encoder for finite values.

    The special NaN code (mantissa 7, exponent 15) is never emitted.  Quantizer
    scale boundary fixtures can replace this helper if a deployed FlashInfer
    revision uses a different scale-rounding policy.
    """
    if x.is_cuda:
        if not x.is_contiguous():
            raise ValueError("CUDA E4M3FN encode requires contiguous storage")
        xf = x.float()
        out = torch.empty(xf.shape, device=xf.device, dtype=torch.uint8)
        encode_e4m3fn_kernel[(triton.cdiv(xf.numel(), 256),)](
            xf, out, numel=xf.numel(), BLOCK=256
        )
        return out
    xf = x.float()
    sign = torch.signbit(xf)
    ax = torch.nan_to_num(xf.abs(), nan=0.0, posinf=E4M3_MAX).clamp_max(E4M3_MAX)
    sub = ax < 2.0**-6
    # Subnormal rounding can carry into the smallest normal (2**-6).
    sub_m_raw = torch.round(ax * 512.0).to(torch.int32)
    sub_m = sub_m_raw.clamp(0, 7)
    exponent_unbiased = torch.floor(torch.log2(ax.clamp_min(2.0**-6))).to(torch.int32)
    mantissa = torch.round((ax / torch.exp2(exponent_unbiased.float()) - 1.0) * 8.0).to(
        torch.int32
    )
    carry = mantissa == 8
    exponent_unbiased = exponent_unbiased + carry.to(torch.int32)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    exponent = (exponent_unbiased + 7).clamp(1, 15)
    # 0x7f is NaN in E4M3FN: saturate finite values at 0x7e (448).
    mantissa = torch.where(exponent == 15, mantissa.clamp_max(6), mantissa)
    encoded_sub = torch.where(sub_m_raw >= 8, torch.full_like(sub_m, 8), sub_m)
    encoded = torch.where(sub, encoded_sub, (exponent << 3) | mantissa)
    encoded = torch.where(ax == 0, torch.zeros_like(encoded), encoded)
    return (encoded | (sign.to(torch.int32) << 7)).to(torch.uint8)


def decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    if codes.is_cuda:
        raw = _as_u8(codes)
        if not raw.is_contiguous():
            raise ValueError("CUDA E2M1 decode requires contiguous storage")
        out = torch.empty(raw.shape, device=raw.device, dtype=torch.float32)
        decode_e2m1_kernel[(triton.cdiv(raw.numel(), 256),)](
            raw, out, numel=raw.numel(), BLOCK=256
        )
        return out
    c = _as_u8(codes).to(torch.int64)
    mag = c & 0x7
    lut = torch.tensor(E2M1_VALUES, device=c.device, dtype=torch.float32)
    value = lut[mag]
    return torch.where((c & 0x8) != 0, -value, value)


def encode_e2m1(x: torch.Tensor) -> torch.Tensor:
    """RNE-even E2M1 encoder, saturated at the finite endpoint +/-6."""
    if x.is_cuda:
        if not x.is_contiguous():
            raise ValueError("CUDA E2M1 encode requires contiguous storage")
        xf = x.float()
        out = torch.empty(xf.shape, device=xf.device, dtype=torch.uint8)
        encode_e2m1_kernel[(triton.cdiv(xf.numel(), 256),)](
            xf, out, numel=xf.numel(), BLOCK=256
        )
        return out
    xf = x.float()
    ax = torch.nan_to_num(xf.abs(), nan=0.0, posinf=6.0).clamp_max(6.0)
    # Midpoints choose the value with even low three-bit code.
    bounds = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    ties_up = (False, True, False, True, False, True, False)
    index = torch.zeros_like(ax, dtype=torch.int32)
    for boundary, up in zip(bounds, ties_up):
        index += (ax >= boundary if up else ax > boundary).to(torch.int32)
    # E2M1 has a signed zero representation.  Keep it: byte fixtures use this
    # to distinguish an IEEE negative zero from a positive underflow.
    signed = index | (torch.signbit(xf).to(torch.int32) << 3)
    return signed.to(torch.uint8)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        raise ValueError("E2M1 packing requires an even K dimension")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    p = _as_u8(packed)
    out = torch.empty(
        (*p.shape[:-1], p.shape[-1] * 2), device=p.device, dtype=torch.uint8
    )
    out[..., 0::2] = p & 0xF
    out[..., 1::2] = p >> 4
    return out


def quantize_fp8(x: torch.Tensor, group_size: int = 128) -> QuantActivation:
    """Dynamic E4M3FN activation quantization with FP32 K-group scales."""
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError(
            "FP8 quantization expects [M,K] with K divisible by group_size"
        )
    m, k = x.shape
    if x.is_cuda:
        if group_size != 128:
            raise ValueError("the V100 Triton FP8 quantizer has fixed group_size=128")
        if not x.is_contiguous():
            raise ValueError("the V100 FP8 quantizer requires contiguous x")
        q = torch.empty((m, k), dtype=torch.uint8, device=x.device)
        scale = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
        if m == 0:
            return QuantActivation("fp8", q, (m, k), scale)
        quantize_fp8_group128_kernel[(m * (k // 128),)](
            x, q, scale, m=m, k=k, BLOCK=128
        )
        return QuantActivation("fp8", q, (m, k), scale)
    # CPU is intentionally a diagnostic fallback; runtime inference always
    # takes the Triton branch above.
    groups = x.float().reshape(m, k // group_size, group_size)
    scale = groups.abs().amax(-1) / E4M3_MAX
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = encode_e4m3fn((groups / safe[..., None]).reshape(m, k))
    return QuantActivation("fp8", q, (m, k), scale.to(torch.float32))


def quantize_nvfp4(
    x: torch.Tensor, global_scale: torch.Tensor | float, group_size: int = 16
) -> QuantActivation:
    """Static-global W4A4 activation quantization in raw linear K16 layout.

    ``global_scale`` is the checkpoint multiplier G, scalar or one scalar per
    row.  No row-amax normalization is performed.
    """
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError("NVFP4 quantization expects [M,K] with K divisible by 16")
    m, k = x.shape
    g = torch.as_tensor(global_scale, dtype=torch.float32, device=x.device)
    if g.numel() == 1:
        g_rows = g.reshape(1, 1)
    elif g.numel() == m:
        g_rows = g.reshape(m, 1)
    else:
        raise ValueError("global_scale must be scalar or have one value per row")
    if x.is_cuda:
        if group_size != 16:
            raise ValueError("the V100 Triton NVFP4 quantizer has fixed group_size=16")
        if not x.is_contiguous():
            raise ValueError("the V100 NVFP4 quantizer requires contiguous x")
        g_kernel = g.contiguous().reshape(-1)
        q = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
        sf = torch.empty((m, k // 16), dtype=torch.uint8, device=x.device)
        quantize_nvfp4_group16_kernel[(m * (k // 16),)](
            x,
            q,
            sf,
            g_kernel,
            m=m,
            k=k,
            GLOBAL_PER_ROW=g_kernel.numel() != 1,
        )
        return QuantActivation("nvfp4", q, (m, k), sf, g)
    # CPU is only a compact diagnostic fallback; model execution uses Triton.
    u = x.float() * g_rows
    blocks = u.reshape(m, k // group_size, group_size)
    raw_sf = blocks.abs().amax(-1) / 6.0
    sf = encode_e4m3fn(raw_sf)
    decoded_sf = decode_e4m3fn(sf)
    safe_sf = torch.where(decoded_sf == 0, torch.ones_like(decoded_sf), decoded_sf)
    q = encode_e2m1((blocks / safe_sf[..., None]).reshape(m, k))
    # A zero local scale represents an all-zero decoded group.  In particular,
    # do not retain an E2M1 negative-zero nibble for a negative-zero input.
    # Standalone E2M1 encoding preserves that representation; the packed
    # NVFP4 payload has this stricter zero-scale contract.
    q = torch.where(
        decoded_sf.repeat_interleave(group_size, dim=-1) == 0,
        torch.zeros_like(q),
        q,
    )
    return QuantActivation("nvfp4", pack_e2m1(q), (m, k), sf, g)


def dequantize_nvfp4(
    packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor | float
) -> torch.Tensor:
    codes = unpack_e2m1(packed)
    sf = decode_e4m3fn(scale).repeat_interleave(16, dim=-1)
    g = torch.as_tensor(global_scale, dtype=torch.float32, device=packed.device)
    if g.numel() == 1:
        g = g.reshape(1, 1)
    else:
        g = g.reshape(-1, 1)
    return (decode_e2m1(codes) * sf / g).to(torch.float16)


def linear_fp8(
    x: torch.Tensor | QuantActivation,
    weight: Weight,
    *,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Raw `[N,K]` checkpoint W8A8 linear using a V100-safe Triton GEMM."""
    qa = x if isinstance(x, QuantActivation) else quantize_fp8(x)
    if qa.kind != "fp8":
        raise ValueError("linear_fp8 requires an FP8 QuantActivation")
    if weight.kind != "fp8":
        raise TypeError("linear_fp8 requires a Weight(kind='fp8')")
    weight_scale = weight.block_scale
    w_data = _as_u8(weight.data)
    if w_data.ndim != 2 or weight_scale is None or weight_scale.ndim != 2:
        raise ValueError("expected raw weight [N,K] and scale [N/128,K/128]")
    m, k = qa.logical_shape
    n = w_data.shape[0]
    if w_data.shape[1] != k or k % 128 or n % 32:
        raise ValueError("V100 FP8 kernel requires K%128==0 and N%32==0")
    if (
        not qa.data.is_cuda
        or qa.data.device != w_data.device
        or qa.block_scale.device != w_data.device
        or weight_scale.device != w_data.device
        or not qa.data.is_contiguous()
        or not qa.block_scale.is_contiguous()
        or not w_data.is_contiguous()
        or not weight_scale.is_contiguous()
    ):
        raise ValueError("linear_fp8 requires contiguous CUDA payloads on one device")
    if m == 0:
        return torch.empty((0, n), device=w_data.device, dtype=output_dtype)
    out = torch.empty((m, n), device=w_data.device, dtype=output_dtype)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    fp8_block128_gemm_kernel[grid](
        qa.data,
        qa.block_scale,
        w_data,
        weight_scale,
        out,
        m=m,
        n=n,
        k=k,
        stride_am=qa.data.stride(0),
        stride_as_m=qa.block_scale.stride(0),
        stride_wn=w_data.stride(0),
        stride_ws_n=weight_scale.stride(0),
        stride_ws_k=weight_scale.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BM=32,
        BN=32,
        BK=128,
        num_warps=4,
        num_stages=1,
    )
    return out
