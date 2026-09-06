from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _decode_e4m3fn(x):
    sign = tl.where((x & 0x80) != 0, -1.0, 1.0)
    exponent = (x >> 3) & 0xF
    mantissa = x & 0x7
    subnormal = mantissa.to(tl.float32) * 0.001953125  # 2**-9
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.math.exp2(
        exponent.to(tl.float32) - 7.0
    )
    value = sign * tl.where(exponent == 0, subnormal, normal)
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), value)


@triton.jit
def _e2m1_decode(x):
    magnitude = x & 7
    value = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(
                            magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((x & 8) != 0, -value, value)


@triton.jit
def _round_even_nonnegative(x):
    """RNE for finite nonnegative values with an explicit tie rule."""
    lower = tl.math.floor(x).to(tl.int32)
    fraction = x - lower.to(tl.float32)
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower & 1) != 0))
    return lower + increment.to(tl.int32)


@triton.jit
def _encode_e2m1(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    # RNE-even at the seven E2M1 midpoints.
    idx = (ax > 0.25).to(tl.int32)
    idx += (ax >= 0.75).to(tl.int32)
    idx += (ax > 1.25).to(tl.int32)
    idx += (ax >= 1.75).to(tl.int32)
    idx += (ax > 2.5).to(tl.int32)
    idx += (ax >= 3.5).to(tl.int32)
    idx += (ax > 5.0).to(tl.int32)
    sign = (x.to(tl.int32, bitcast=True) & 0x80000000) != 0
    return idx | sign.to(tl.int32) * 8


@triton.jit
def _encode_e4m3fn(x):
    """Finite RNE/saturating E4M3FN software encoder."""
    ax = tl.minimum(tl.abs(x), 448.0)
    is_sub = ax < 0.015625
    sub_m = _round_even_nonnegative(ax * 512.0)
    sub_code = tl.minimum(sub_m, 8)
    e = tl.math.floor(tl.math.log2(tl.maximum(ax, 0.015625))).to(tl.int32)
    m = _round_even_nonnegative((ax * tl.math.exp2(-e.to(tl.float32)) - 1.0) * 8.0)
    carry = m == 8
    e += carry.to(tl.int32)
    m = tl.where(carry, 0, m)
    eb = tl.maximum(1, tl.minimum(e + 7, 15))
    m = tl.where(eb == 15, tl.minimum(m, 6), m)
    normal = (eb << 3) | m
    code = tl.where(is_sub, sub_code, normal)
    code = tl.where(ax == 0.0, 0, code)
    sign = (x.to(tl.int32, bitcast=True) & 0x80000000) != 0
    return code | sign.to(tl.int32) * 128


@triton.jit
def encode_e2m1_kernel(x_ptr, out_ptr, numel: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < numel, other=0.0)
    tl.store(out_ptr + offsets, _encode_e2m1(x), mask=offsets < numel)


@triton.jit
def decode_e2m1_kernel(x_ptr, out_ptr, numel: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < numel, other=0).to(tl.int32)
    tl.store(out_ptr + offsets, _e2m1_decode(x), mask=offsets < numel)


@triton.jit
def encode_e4m3fn_kernel(x_ptr, out_ptr, numel: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < numel, other=0.0)
    tl.store(out_ptr + offsets, _encode_e4m3fn(x), mask=offsets < numel)


@triton.jit
def decode_e4m3fn_kernel(x_ptr, out_ptr, numel: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < numel, other=0).to(tl.int32)
    tl.store(out_ptr + offsets, _decode_e4m3fn(x), mask=offsets < numel)


@triton.jit
def quantize_fp8_group128_kernel(
    x_ptr, q_ptr, sf_ptr, m: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // (k // BLOCK)
    group = pid % (k // BLOCK)
    offs = group * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * k + offs, mask=row < m, other=0.0).to(tl.float32)
    # Freeze scale formation as RN32(max_abs * RN32(1/448)); CUDA PyTorch's
    # reference codec uses this reciprocal-multiply path.  Payload division
    # below is explicitly IEEE RN to protect E4M3 midpoint decisions.
    scale = tl.max(tl.abs(x), axis=0) / 448.0
    safe = tl.where(scale == 0.0, 1.0, scale)
    tl.store(
        q_ptr + row * k + offs, _encode_e4m3fn(tl.math.div_rn(x, safe)), mask=row < m
    )
    tl.store(sf_ptr + row * (k // BLOCK) + group, scale, mask=row < m)


@triton.jit
def quantize_nvfp4_group16_kernel(
    x_ptr,
    q_ptr,
    sf_ptr,
    global_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    GLOBAL_PER_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // (k // 16)
    group = pid % (k // 16)
    pair = tl.arange(0, 8)
    offs = group * 16 + pair * 2
    g = tl.load(global_ptr + tl.where(GLOBAL_PER_ROW, row, 0)).to(tl.float32)
    u_lo = tl.load(x_ptr + row * k + offs, mask=row < m, other=0.0).to(tl.float32) * g
    u_hi = (
        tl.load(x_ptr + row * k + offs + 1, mask=row < m, other=0.0).to(tl.float32) * g
    )
    # As above, retain the reference's reciprocal-multiply scale formation.
    raw_sf = (
        tl.maximum(tl.max(tl.abs(u_lo), axis=0), tl.max(tl.abs(u_hi), axis=0)) / 6.0
    )
    sf = _encode_e4m3fn(raw_sf)
    decoded = _decode_e4m3fn(sf)
    safe = tl.where(decoded == 0.0, 1.0, decoded)
    lo_code = _encode_e2m1(tl.math.div_rn(u_lo, safe)).to(tl.uint8)
    hi_code = _encode_e2m1(tl.math.div_rn(u_hi, safe)).to(tl.uint8)
    # A zero E4M3 local scale has an all-zero packed E2M1 payload contract.
    lo_code = tl.where(sf == 0, 0, lo_code)
    hi_code = tl.where(sf == 0, 0, hi_code)
    packed = lo_code | (hi_code << 4)
    tl.store(q_ptr + row * (k // 2) + group * 8 + pair, packed, mask=row < m)
    tl.store(sf_ptr + row * (k // 16) + group, sf, mask=row < m)


@triton.jit
def fp8_block128_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    w_ptr,
    w_scale_ptr,
    out_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_as_m: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_ws_n: tl.constexpr,
    stride_ws_k: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """E4M3FN A8/W8 GEMM with FP32 block-scale accumulation.

    BK must divide 128: an activation and weight scale is constant for each
    K=128 region, but deliberately remains FP32 rather than being folded into
    an FP16 operand.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    # Volta's Triton 2.3 MMA lowering accepts K=32 tiles.  The four dot
    # products below form one complete K=128 scale region before its FP32
    # activation/weight scale is applied.
    for k0 in range(0, k, 128):
        part = tl.zeros((BM, BN), tl.float32)
        for kk in range(0, 128, 32):
            offs_k = k0 + kk + tl.arange(0, 32)
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0,
            )
            w = tl.load(
                w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None],
                mask=(offs_n[None, :] < n) & (offs_k[:, None] < k),
                other=0,
            )
            part += tl.dot(
                _decode_e4m3fn(a).to(tl.float16),
                _decode_e4m3fn(w).to(tl.float16),
            )
        group_k = k0 // 128
        ascale = tl.load(
            a_scale_ptr + offs_m * stride_as_m + group_k,
            mask=offs_m < m,
            other=0.0,
        )
        wscale = tl.load(
            w_scale_ptr + (offs_n // 128) * stride_ws_n + group_k * stride_ws_k,
            mask=offs_n < n,
            other=0.0,
        )
        acc += part * ascale[:, None] * wscale[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )
