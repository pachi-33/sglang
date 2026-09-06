from __future__ import annotations

import triton
import triton.language as tl

from .quantization import _decode_e4m3fn, _encode_e2m1, _encode_e4m3fn


@triton.jit
def dispatch_count_kernel(
    ids_ptr,
    counts_ptr,
    routes: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    count = 0
    for start in range(0, ROUTE_BLOCK, BLOCK):
        route = start + tl.arange(0, BLOCK)
        ids = tl.load(ids_ptr + route, mask=route < routes, other=-1)
        count += tl.sum((ids == expert).to(tl.int32), axis=0)
    tl.store(counts_ptr + expert, count)


@triton.jit
def dispatch_prefix_kernel(counts_ptr, offsets_ptr):
    expert = tl.arange(0, 256)
    counts = tl.load(counts_ptr + expert)
    blocks = (counts + 31) // 32
    inclusive = tl.cumsum(blocks, axis=0)
    tl.store(offsets_ptr + expert, inclusive - blocks)
    tl.store(offsets_ptr + 256, tl.sum(blocks, axis=0))


@triton.jit
def dispatch_init_kernel(
    source_ptr, positions_ptr, capacity: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(source_ptr + offsets, -1, mask=offsets < capacity)
    tl.store(positions_ptr + offsets, -1, mask=offsets < capacity)


@triton.jit
def dispatch_block_expert_kernel(
    offsets_ptr, block_expert_ptr, max_blocks: tl.constexpr
):
    block = tl.program_id(0)
    expert = tl.arange(0, 256)
    start = tl.load(offsets_ptr + expert)
    end = tl.load(offsets_ptr + expert + 1)
    selected = tl.min(tl.where((block >= start) & (block < end), expert, 256), axis=0)
    tl.store(
        block_expert_ptr + block,
        tl.where(selected == 256, -1, selected),
        mask=block < max_blocks,
    )


@triton.jit
def dispatch_stable_scatter_kernel(
    ids_ptr,
    source_ptr,
    positions_ptr,
    offsets_ptr,
    routes: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    destination = tl.load(offsets_ptr + expert) * 32
    running = 0
    # One program per expert walks route chunks in source order.  The running
    # count is a stable per-expert parallel scan, O(E*R) total work.
    for start in range(0, ROUTE_BLOCK, BLOCK):
        route = start + tl.arange(0, BLOCK)
        ids = tl.load(ids_ptr + route, mask=route < routes, other=-1)
        selected = ids == expert
        rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
        target = destination + running + rank
        tl.store(source_ptr + target, route // 8, mask=(route < routes) & selected)
        tl.store(positions_ptr + target, route, mask=(route < routes) & selected)
        running += tl.sum(selected.to(tl.int32), axis=0)


@triton.jit
def dispatch_inverse_kernel(
    positions_ptr,
    inverse_ptr,
    rows: tl.constexpr,
    routes: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    position = tl.load(positions_ptr + row, mask=row < rows, other=-1)
    tl.store(
        inverse_ptr + position,
        row,
        mask=(row < rows) & (position >= 0) & (position < routes),
    )


@triton.jit
def route_combine_kernel(
    down_ptr,
    inverse_ptr,
    weights_ptr,
    out_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    stride_down_m: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in range(0, 8):
        row = tl.load(inverse_ptr + token * 8 + slot, mask=token < tokens, other=-1)
        safe_row = tl.where(row >= 0, row, 0)
        value = tl.load(
            down_ptr + safe_row * stride_down_m + cols,
            mask=(row >= 0) & (cols < hidden),
            other=0.0,
        )
        weight = tl.load(weights_ptr + token * 8 + slot, mask=token < tokens, other=0.0)
        acc += value.to(tl.float32) * weight
    tl.store(
        out_ptr + token * hidden + cols, acc, mask=(token < tokens) & (cols < hidden)
    )


@triton.jit
def normalized_top8_kernel(
    logits_ptr,
    ids_ptr,
    raw_weights_ptr,
    tokens: tl.constexpr,
    experts: tl.constexpr,
    TOPK: tl.constexpr,
):
    """One CTA per token: stable FP32 softmax and exact lower-ID tie breaks."""
    token = tl.program_id(0)
    expert = tl.arange(0, 256)
    logits = tl.load(
        logits_ptr + token * experts + expert,
        mask=(token < tokens) & (expert < experts),
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(logits, axis=0)
    probability = tl.exp(logits - maximum)
    probability /= tl.sum(probability, axis=0)
    score = probability
    for slot in range(0, TOPK):
        best = tl.max(score, axis=0)
        # Min over the matching expert IDs makes equal router values stable.
        chosen = tl.min(tl.where(score == best, expert, experts), axis=0)
        selected = expert == chosen
        p = tl.sum(tl.where(selected, probability, 0.0), axis=0)
        tl.store(ids_ptr + token * TOPK + slot, chosen, mask=token < tokens)
        tl.store(raw_weights_ptr + token * TOPK + slot, p, mask=token < tokens)
        score = tl.where(selected, -float("inf"), score)


@triton.jit
def normalize_topk_weights_kernel(
    weights_ptr, tokens: tl.constexpr, TOPK: tl.constexpr
):
    token = tl.program_id(0)
    slot = tl.arange(0, TOPK)
    weights = tl.load(
        weights_ptr + token * TOPK + slot, mask=token < tokens, other=0.0
    ).to(tl.float32)
    total = tl.sum(weights, axis=0)
    tl.store(weights_ptr + token * TOPK + slot, weights / total, mask=token < tokens)


@triton.jit
def _e2m1(code):
    mag = code & 7
    value = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(
                        mag == 4,
                        2.0,
                        tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _e4m3fn(byte):
    sign = tl.where((byte & 128) != 0, -1.0, 1.0)
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    subnormal = mantissa.to(tl.float32) * 0.001953125
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.math.exp2(
        exponent.to(tl.float32) - 7.0
    )
    return sign * tl.where(exponent == 0, subnormal, normal)


@triton.jit
def gather_nvfp4_activation_kernel(
    src_data_ptr,
    src_sf_ptr,
    source_ids_ptr,
    dst_data_ptr,
    dst_sf_ptr,
    rows: tl.constexpr,
    half_k: tl.constexpr,
    sf_k: tl.constexpr,
    stride_src_m: tl.constexpr,
    stride_src_sf_m: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather raw activation rows into expert-major contiguous storage."""
    row = tl.program_id(0)
    source = tl.load(source_ids_ptr + row, mask=row < rows, other=-1)
    valid = source >= 0
    safe_source = tl.where(valid, source, 0)
    offsets = tl.arange(0, BLOCK)
    data = tl.load(
        src_data_ptr + safe_source * stride_src_m + offsets,
        mask=valid & (offsets < half_k),
        other=0,
    )
    sf = tl.load(
        src_sf_ptr + safe_source * stride_src_sf_m + offsets,
        mask=valid & (offsets < sf_k),
        other=0,
    )
    tl.store(dst_data_ptr + row * half_k + offsets, data, mask=offsets < half_k)
    tl.store(dst_sf_ptr + row * sf_k + offsets, sf, mask=offsets < sf_k)


@triton.jit
def expert_row_global_kernel(
    expert_of_block_ptr,
    expert_global_ptr,
    row_global_ptr,
    rows: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(expert_of_block_ptr + row // 32, mask=row < rows, other=-1)
    value = tl.load(
        expert_global_ptr + tl.maximum(expert, 0),
        mask=(row < rows) & (expert >= 0),
        other=0.0,
    )
    tl.store(row_global_ptr + row, value, mask=row < rows)


@triton.jit
def nvfp4_grouped_gemm_kernel(
    a_ptr,
    a_sf_ptr,
    expert_of_block_ptr,
    w_ptr,
    w_sf_ptr,
    a_global_ptr,
    w_global_ptr,
    out_ptr,
    positions_ptr,
    route_weights_ptr,
    rows: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_as_m: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_ws_e: tl.constexpr,
    stride_ws_n: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WEIGHTED: tl.constexpr,
):
    """Raw E2M1/E4M3 grouped GEMM; rows are expert-padded in BM groups."""
    tl.static_assert(BM == 32)
    tl.static_assert(BN == 32)
    tl.static_assert(BK == 32)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    raw_expert = tl.load(
        expert_of_block_ptr + pid_m, mask=pid_m < (rows // BM), other=-1
    )
    valid_expert = raw_expert >= 0
    expert = tl.where(valid_expert, raw_expert, 0)
    if valid_expert:
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, k, BK):
            packed_k = k0 // 2 + tl.arange(0, 16)
            offs_k = packed_k * 2
            a_byte = tl.load(
                a_ptr + offs_m[:, None] * stride_am + packed_k[None, :],
                mask=(offs_m[:, None] < rows) & (offs_k[None, :] < k),
                other=0,
            )
            a_lo_code, a_hi_code = a_byte & 15, a_byte >> 4
            w_byte = tl.load(
                w_ptr
                + expert * stride_we
                + offs_n[None, :] * stride_wn
                + packed_k[:, None],
                mask=(offs_n[None, :] < n) & (offs_k[:, None] < k),
                other=0,
            )
            w_lo_code, w_hi_code = w_byte & 15, w_byte >> 4
            a_sf = tl.load(
                a_sf_ptr + offs_m[:, None] * stride_as_m + (offs_k[None, :] // 16),
                mask=(offs_m[:, None] < rows) & (offs_k[None, :] < k),
                other=0,
            )
            w_sf = tl.load(
                w_sf_ptr
                + expert * stride_ws_e
                + offs_n[None, :] * stride_ws_n
                + (offs_k[:, None] // 16),
                mask=(offs_n[None, :] < n) & (offs_k[:, None] < k),
                other=0,
            )
            a_lo = (_e2m1(a_lo_code) * _e4m3fn(a_sf)).to(tl.float16)
            a_hi = (_e2m1(a_hi_code) * _e4m3fn(a_sf)).to(tl.float16)
            w_lo = (_e2m1(w_lo_code) * _e4m3fn(w_sf)).to(tl.float16)
            w_hi = (_e2m1(w_hi_code) * _e4m3fn(w_sf)).to(tl.float16)
            acc += tl.dot(a_lo, w_lo)
            acc += tl.dot(a_hi, w_hi)
        out = acc * (
            1.0 / (tl.load(a_global_ptr + expert) * tl.load(w_global_ptr + expert))
        )
    else:
        out = tl.zeros((BM, BN), tl.float32)
    store_mask = valid_expert & (offs_m[:, None] < rows) & (offs_n[None, :] < n)
    if WEIGHTED:
        route = tl.load(positions_ptr + offs_m, mask=valid_expert, other=-1)
        route_weight = tl.load(
            route_weights_ptr + tl.maximum(route, 0)[:, None],
            mask=valid_expert & (route[:, None] >= 0),
            other=0.0,
        )
        value = out.to(tl.float16).to(tl.float32) * route_weight
    else:
        value = out
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        value,
        mask=store_mask,
    )


@triton.jit
def route_combine_weighted_kernel(
    down_ptr,
    inverse_ptr,
    out_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    stride_down_m: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in range(0, 8):
        row = tl.load(inverse_ptr + token * 8 + slot, mask=token < tokens, other=-1)
        acc += tl.load(
            down_ptr + tl.maximum(row, 0) * stride_down_m + cols,
            mask=(row >= 0) & (cols < hidden),
            other=0.0,
        )
    tl.store(
        out_ptr + token * hidden + cols, acc, mask=(token < tokens) & (cols < hidden)
    )


@triton.jit
def route_combine_fused_kernel(
    down_ptr,
    inverse_ptr,
    out_ptr,
    shared_ptr,
    residual_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    stride_down_m: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in range(0, 8):
        row = tl.load(inverse_ptr + token * 8 + slot, mask=token < tokens, other=-1)
        acc += tl.load(
            down_ptr + tl.maximum(row, 0) * stride_down_m + cols,
            mask=(row >= 0) & (cols < hidden),
            other=0.0,
        )
    result = acc.to(tl.float16)
    if HAS_SHARED:
        shared = tl.load(
            shared_ptr + token * hidden + cols,
            mask=(token < tokens) & (cols < hidden),
            other=0.0,
        )
        result = (result.to(tl.float32) + shared.to(tl.float32)).to(tl.float16)
    if HAS_RESIDUAL:
        residual = tl.load(
            residual_ptr + token * hidden + cols,
            mask=(token < tokens) & (cols < hidden),
            other=0.0,
        )
        result = (result.to(tl.float32) + residual.to(tl.float32)).to(tl.float16)
    tl.store(
        out_ptr + token * hidden + cols, result, mask=(token < tokens) & (cols < hidden)
    )


@triton.jit
def route_weight_fp16_kernel(
    down_ptr,
    positions_ptr,
    weights_ptr,
    out_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    stride_down_m: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    position = tl.load(positions_ptr + row, mask=row < rows, other=-1)
    value = tl.load(
        down_ptr + row * stride_down_m + cols,
        mask=(row < rows) & (position >= 0) & (cols < hidden),
        other=0.0,
    )
    weight = tl.load(
        weights_ptr + tl.maximum(position, 0),
        mask=(row < rows) & (position >= 0),
        other=0.0,
    )
    tl.store(
        out_ptr + row * hidden + cols,
        value.to(tl.float32) * weight,
        mask=(row < rows) & (position >= 0) & (cols < hidden),
    )


@triton.jit
def nvfp4_paired_gemm1_swiglu_a4_kernel(
    a_ptr,
    a_sf_ptr,
    source_ids_ptr,
    expert_of_block_ptr,
    w_ptr,
    w_sf_ptr,
    gate_a_global_ptr,
    gate_w_global_ptr,
    down_a_global_ptr,
    scratch_ptr,
    capture_z_ptr,
    z_ptr,
    z_sf_ptr,
    rows: tl.constexpr,
    intermediate: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_as_m: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_ws_e: tl.constexpr,
    stride_ws_n: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    MAX_TILES_PER_PROGRAM: tl.constexpr,
    CAPTURE_Z: tl.constexpr,
):
    """Paired raw-NVFP4 GEMM1, FP16 SwiGLU boundary, and fused A4 encode.

    One persistent CTA owns its 32x32 FP16 scratch tile.  The scratch is only
    used to cross the Volta Triton MMA/reduction layout boundary; it is reused
    for every tile assigned to that CTA and never shared between CTAs.
    """
    tl.static_assert(intermediate % 32 == 0)
    tl.static_assert(k % 32 == 0)
    pid = tl.program_id(0)
    rows32 = tl.arange(0, 32)
    cols32 = tl.arange(0, 32)
    n_tiles = intermediate // 32
    scratch = scratch_ptr + pid * 1024

    for work in range(0, MAX_TILES_PER_PROGRAM):
        tile = pid + work * NUM_PROGRAMS
        valid_tile = tile < (rows // 32) * n_tiles
        block = tile // n_tiles
        n_tile = tile - block * n_tiles
        offs_m = block * 32 + rows32
        offs_n = n_tile * 32 + cols32
        raw_expert = tl.load(
            expert_of_block_ptr + block,
            mask=valid_tile & (block < rows // 32),
            other=-1,
        )
        valid_expert = valid_tile & (raw_expert >= 0)
        expert = tl.where(valid_expert, raw_expert, 0)
        source = tl.load(source_ids_ptr + offs_m, mask=valid_expert, other=-1)
        valid_source = source >= 0
        safe_source = tl.where(valid_source, source, 0)
        # Dispatch capacity includes invalid tail blocks.  The scalar branch
        # prevents those CTAs from issuing K-loop memory traffic or HMMA work.
        if valid_expert:
            gate_acc = tl.zeros((32, 32), tl.float32)
            up_acc = tl.zeros((32, 32), tl.float32)
            # Keep the two packed nibbles as distinct K16 MMA operands.
            for k0 in range(0, k, 32):
                packed_k = k0 // 2 + tl.arange(0, 16)
                offs_k = packed_k * 2
                a_byte = tl.load(
                    a_ptr + safe_source[:, None] * stride_am + packed_k[None, :],
                    mask=valid_source[:, None] & (offs_k[None, :] < k),
                    other=0,
                )
                a_sf = tl.load(
                    a_sf_ptr
                    + safe_source[:, None] * stride_as_m
                    + (offs_k[None, :] // 16),
                    mask=valid_source[:, None] & (offs_k[None, :] < k),
                    other=0,
                )
                a_lo = (_e2m1(a_byte & 15) * _e4m3fn(a_sf)).to(tl.float16)
                a_hi = (_e2m1(a_byte >> 4) * _e4m3fn(a_sf)).to(tl.float16)
                gate_byte = tl.load(
                    w_ptr
                    + expert * stride_we
                    + offs_n[None, :] * stride_wn
                    + packed_k[:, None],
                    mask=offs_k[:, None] < k,
                    other=0,
                )
                up_byte = tl.load(
                    w_ptr
                    + expert * stride_we
                    + (offs_n[None, :] + intermediate) * stride_wn
                    + packed_k[:, None],
                    mask=offs_k[:, None] < k,
                    other=0,
                )
                gate_sf = tl.load(
                    w_sf_ptr
                    + expert * stride_ws_e
                    + offs_n[None, :] * stride_ws_n
                    + (offs_k[:, None] // 16),
                    mask=offs_k[:, None] < k,
                    other=0,
                )
                up_sf = tl.load(
                    w_sf_ptr
                    + expert * stride_ws_e
                    + (offs_n[None, :] + intermediate) * stride_ws_n
                    + (offs_k[:, None] // 16),
                    mask=offs_k[:, None] < k,
                    other=0,
                )
                gate_lo = (_e2m1(gate_byte & 15) * _e4m3fn(gate_sf)).to(tl.float16)
                gate_hi = (_e2m1(gate_byte >> 4) * _e4m3fn(gate_sf)).to(tl.float16)
                up_lo = (_e2m1(up_byte & 15) * _e4m3fn(up_sf)).to(tl.float16)
                up_hi = (_e2m1(up_byte >> 4) * _e4m3fn(up_sf)).to(tl.float16)
                gate_acc += tl.dot(a_lo, gate_lo) + tl.dot(a_hi, gate_hi)
                up_acc += tl.dot(a_lo, up_lo) + tl.dot(a_hi, up_hi)
            inv_global = 1.0 / (
                tl.load(gate_a_global_ptr + expert).to(tl.float32)
                * tl.load(gate_w_global_ptr + expert).to(tl.float32)
            )
            gate = (gate_acc * inv_global).to(tl.float16)
            up = (up_acc * inv_global).to(tl.float16)
            z = (
                gate.to(tl.float32)
                * tl.sigmoid(gate.to(tl.float32))
                * up.to(tl.float32)
            ).to(tl.float16)
        else:
            z = tl.zeros((32, 32), tl.float16)
        tl.store(scratch + rows32[:, None] * 32 + cols32[None, :], z)
        if CAPTURE_Z:
            tl.store(
                capture_z_ptr + offs_m[:, None] * intermediate + offs_n[None, :],
                z,
                mask=valid_expert,
            )
        tl.debug_barrier()

        # Each group uses independent rank-1 [32,8] volatile loads from the
        # CTA-exclusive scratch tile, avoiding Triton 2.3's MMA-layout to
        # reduction-layout miscompile.
        pair = tl.arange(0, 8)
        for group_in_tile in range(0, 2):
            local = group_in_tile * 16 + pair * 2
            z_lo = tl.load(
                scratch + rows32[:, None] * 32 + local[None, :], volatile=True
            ).to(tl.float32)
            z_hi = tl.load(
                scratch + rows32[:, None] * 32 + local[None, :] + 1, volatile=True
            ).to(tl.float32)
            down_g = tl.load(down_a_global_ptr + expert).to(tl.float32)
            u_lo = z_lo * down_g
            u_hi = z_hi * down_g
            raw_sf = (
                tl.maximum(tl.max(tl.abs(u_lo), axis=1), tl.max(tl.abs(u_hi), axis=1))
                / 6.0
            )
            sf = _encode_e4m3fn(raw_sf)
            decoded = _decode_e4m3fn(sf)
            safe = tl.where(decoded == 0.0, 1.0, decoded)
            lo_code = _encode_e2m1(tl.math.div_rn(u_lo, safe[:, None])).to(tl.uint8)
            hi_code = _encode_e2m1(tl.math.div_rn(u_hi, safe[:, None])).to(tl.uint8)
            lo_code = tl.where(sf[:, None] == 0, 0, lo_code)
            hi_code = tl.where(sf[:, None] == 0, 0, hi_code)
            packed = lo_code | (hi_code << 4)
            z_group = n_tile * 2 + group_in_tile
            tl.store(
                z_ptr
                + offs_m[:, None] * (intermediate // 2)
                + z_group * 8
                + pair[None, :],
                packed,
                mask=valid_expert,
            )
            tl.store(
                z_sf_ptr + offs_m * (intermediate // 16) + z_group,
                sf,
                mask=valid_expert,
            )
        # Ensure all volatile consumers have finished before this CTA writes
        # its scratch slot for the next persistent tile.
        tl.debug_barrier()


@triton.jit
def fp16_grouped_gemm_kernel(
    a_ptr,
    source_ids_ptr,
    expert_of_block_ptr,
    w_ptr,
    out_ptr,
    rows: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    IDENTITY_SOURCE: tl.constexpr,
):
    """Expert-major FP16 NT GEMM with a token/source indirection."""
    block = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = block * 32 + tl.arange(0, 32)
    offs_n = pid_n * 32 + tl.arange(0, 32)
    raw_expert = tl.load(expert_of_block_ptr + block, mask=block < rows // 32, other=-1)
    valid_expert = raw_expert >= 0
    expert = tl.where(valid_expert, raw_expert, 0)
    if IDENTITY_SOURCE:
        source = offs_m
    else:
        source = tl.load(source_ids_ptr + offs_m, mask=valid_expert, other=-1)
    valid_source = source >= 0
    safe_source = tl.where(valid_source, source, 0)
    acc = tl.zeros((32, 32), tl.float32)
    for k0 in range(0, k, 32):
        offs_k = k0 + tl.arange(0, 32)
        a = tl.load(
            a_ptr + safe_source[:, None] * stride_am + offs_k[None, :],
            mask=valid_expert & valid_source[:, None] & (offs_k[None, :] < k),
            other=0.0,
        )
        w = tl.load(
            w_ptr + expert * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None],
            mask=valid_expert & (offs_n[None, :] < n) & (offs_k[:, None] < k),
            other=0.0,
        )
        acc += tl.dot(a, w)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=valid_expert & (offs_n[None, :] < n),
    )


@triton.jit
def fp16_swiglu_kernel(
    gate_up_ptr,
    z_ptr,
    rows: tl.constexpr,
    intermediate: tl.constexpr,
    stride_gu_m: tl.constexpr,
    stride_z_m: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    gate = tl.load(
        gate_up_ptr + row * stride_gu_m + cols,
        mask=(row < rows) & (cols < intermediate),
        other=0.0,
    )
    up = tl.load(
        gate_up_ptr + row * stride_gu_m + intermediate + cols,
        mask=(row < rows) & (cols < intermediate),
        other=0.0,
    )
    z = (gate.to(tl.float32) * tl.sigmoid(gate.to(tl.float32)) * up.to(tl.float32)).to(
        tl.float16
    )
    tl.store(
        z_ptr + row * stride_z_m + cols, z, mask=(row < rows) & (cols < intermediate)
    )


@triton.jit
def fp16_sigmoid_multiply_kernel(
    value_ptr,
    gate_ptr,
    out_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    stride_value_m: tl.constexpr,
    stride_gate_m: tl.constexpr,
    stride_out_m: tl.constexpr,
    GATE_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        value_ptr + row * stride_value_m + cols,
        mask=(row < rows) & (cols < hidden),
        other=0.0,
    )
    gate = tl.load(
        gate_ptr + row * stride_gate_m + tl.where(GATE_SCALAR, 0, cols),
        mask=(row < rows) & (cols < hidden),
        other=0.0,
    )
    # CUDA's shared-expert path rounds sigmoid(gate_fp16) to FP16 before the
    # FP32 multiply with the FP16 shared output.
    gate_sigmoid = tl.sigmoid(gate.to(tl.float32)).to(tl.float16)
    out = (value.to(tl.float32) * gate_sigmoid.to(tl.float32)).to(tl.float16)
    tl.store(
        out_ptr + row * stride_out_m + cols, out, mask=(row < rows) & (cols < hidden)
    )


@triton.jit
def fp16_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    stride_a_m: tl.constexpr,
    stride_b_m: tl.constexpr,
    stride_out_m: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(
        a_ptr + row * stride_a_m + cols, mask=(row < rows) & (cols < hidden), other=0.0
    )
    b = tl.load(
        b_ptr + row * stride_b_m + cols, mask=(row < rows) & (cols < hidden), other=0.0
    )
    tl.store(
        out_ptr + row * stride_out_m + cols,
        (a.to(tl.float32) + b.to(tl.float32)).to(tl.float16),
        mask=(row < rows) & (cols < hidden),
    )
