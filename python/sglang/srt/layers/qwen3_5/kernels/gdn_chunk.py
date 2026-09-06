"""BT16 stateful Gated DeltaNet Triton launcher for SM70.

The ordered launch is deliberately one block at a time: it establishes the
cross-block state dependency without host reads of packed sequence metadata.
"""

import triton
import triton.language as tl


@triton.jit
def _kkt16(q, k, cu, gram, CHUNKS: tl.constexpr, token_offset):
    """One isolated 16x128 by 128x16 Tensor Core product per Q/K head."""
    seq, chunk, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p, d = tl.arange(0, 16), tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    acc = tl.zeros((16, 16), tl.float32)
    for offset in range(0, 128, 16):
        lhs = tl.load(
            k + (t[:, None] * 16 + h // 2) * 128 + offset + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        rhs = tl.load(
            k + (t[None, :] * 16 + h // 2) * 128 + offset + d[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        acc += tl.dot(lhs, rhs)
    base = ((seq * CHUNKS + chunk) * 32 + h) * 256 + p[:, None] * 16 + p[None, :]
    tl.store(gram + base, acc, mask=valid[:, None] & valid[None, :])


@triton.jit
def _inverse16_col_f32(gram, decay, beta, cu, a16, CHUNKS: tl.constexpr, token_offset):
    """One A column: rank-1 FP32 forward solve and one final FP16 store."""
    seq, chunk, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h, col = tile // 16, tile % 16
    p = tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    g = tl.cumsum(
        tl.load(decay + t * 32 + h, mask=valid, other=0.0).to(tl.float32), axis=0
    )
    base = ((seq * CHUNKS + chunk) * 32 + h) * 256
    acol = tl.zeros((16,), tl.float32)
    for row in range(0, 16):
        valid_row = start + row < end
        gi = tl.sum(
            tl.where(
                p <= row,
                tl.load(decay + t * 32 + h, mask=valid, other=0.0).to(tl.float32),
                0.0,
            ),
            axis=0,
        )
        bi = tl.load(beta + (start + row) * 32 + h, mask=valid_row, other=0.0).to(
            tl.float32
        )
        lower = (p < row) & valid & valid_row
        lrow = tl.where(
            lower,
            bi
            * tl.load(gram + base + row * 16 + p, mask=lower, other=0.0)
            * tl.exp(tl.where(lower, gi - g, 0.0)),
            0.0,
        )
        value = (col == row).to(tl.float32) - tl.sum(lrow * acol, axis=0)
        acol = tl.where(p == row, value, acol)
    tl.store(a16 + base + p * 16 + col, acol.to(tl.float16))


def compute_gram_a16(q, k, decay, beta, cu, max_seqlen):
    """Return separated BT16 KKT (FP32) and A (FP16) scratch tensors."""
    import torch

    batches, chunks = cu.numel() - 1, triton.cdiv(max_seqlen, 16)
    shape = (batches, chunks, 32, 16, 16)
    gram = torch.zeros(shape, device=q.device, dtype=torch.float32)
    a = torch.empty(shape, device=q.device, dtype=torch.float16)
    if chunks:
        _kkt16[(batches, chunks, 32)](
            q, k, cu, gram, CHUNKS=chunks, token_offset=0, num_warps=4, num_stages=1
        )
        _inverse16_col_f32[(batches, chunks, 512)](
            gram,
            decay,
            beta,
            cu,
            a,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
    return gram, a


@triton.jit
def _wy_u16(v, beta, cu, a16, u16, CHUNKS: tl.constexpr, token_offset):
    """One 16x16x16 A*betaV Tensor Core tile; no dependent MMA follows."""
    block, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, chunk = block // CHUNKS, block % CHUNKS
    p, d = tl.arange(0, 16), tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    mbase = ((seq * CHUNKS + chunk) * 32 + h) * 256
    obase = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    aa = tl.load(a16 + mbase + p[:, None] * 16 + p[None, :])
    vv = tl.load(
        v + (t[:, None] * 32 + h) * 128 + tile * 16 + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    bb = tl.load(beta + t * 32 + h, mask=valid, other=0.0).to(tl.float32)
    scaled = (bb[:, None] * vv.to(tl.float32)).to(tl.float16)
    tl.store(
        u16 + obase + p[:, None] * 128 + tile * 16 + d[None, :],
        tl.dot(aa, scaled).to(tl.float16),
        mask=valid[:, None],
    )


@triton.jit
def _prefix_g16_grid(decay, cu, g16, CHUNKS: tl.constexpr, token_offset):
    block, h = tl.program_id(0), tl.program_id(1)
    seq, chunk = block // CHUNKS, block % CHUNKS
    p = tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    g = tl.cumsum(
        tl.load(decay + t * 32 + h, mask=valid, other=0.0).to(tl.float32), axis=0
    )
    base = ((seq * CHUNKS + chunk) * 32 + h) * 16 + p
    tl.store(g16 + base, g, mask=valid)


@triton.jit
def _prefix_g16(decay, cu, g16, chunk, CHUNKS: tl.constexpr, token_offset):
    """Runtime-index form used by state/output's ordered chunk launches."""
    seq, h = tl.program_id(0), tl.program_id(1)
    p = tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    g = tl.cumsum(
        tl.load(decay + t * 32 + h, mask=valid, other=0.0).to(tl.float32), axis=0
    )
    base = ((seq * CHUNKS + chunk) * 32 + h) * 16 + p
    tl.store(g16 + base, g, mask=valid)


@triton.jit
def _scale_w16(k, beta, cu, g16, wk16, CHUNKS: tl.constexpr, token_offset):
    """Materialize the FP16 beta*exp(G)*K boundary before the W MMA."""
    block, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, chunk = block // CHUNKS, block % CHUNKS
    p, d = tl.arange(0, 16), tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    obase = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    kk = tl.load(
        k + (t[:, None] * 16 + h // 2) * 128 + tile * 16 + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    gg = tl.load(
        g16 + ((seq * CHUNKS + chunk) * 32 + h) * 16 + p, mask=valid, other=0.0
    )
    bb = tl.load(beta + t * 32 + h, mask=valid, other=0.0).to(tl.float32)
    scaled = (bb[:, None] * tl.exp(gg)[:, None] * kk.to(tl.float32)).to(tl.float16)
    tl.store(
        wk16 + obase + p[:, None] * 128 + tile * 16 + d[None, :],
        scaled,
        mask=valid[:, None],
    )


@triton.jit
def _wy_w16(wk16, a16, cu, w16, CHUNKS: tl.constexpr, token_offset):
    """One 16x16x16 A*scaled-K Tensor Core tile with no layout conversion."""
    block, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, chunk = block // CHUNKS, block % CHUNKS
    p, d = tl.arange(0, 16), tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    mbase = ((seq * CHUNKS + chunk) * 32 + h) * 256
    obase = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    aa = tl.load(a16 + mbase + p[:, None] * 16 + p[None, :])
    scaled = tl.load(
        wk16 + obase + p[:, None] * 128 + tile * 16 + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    tl.store(
        w16 + obase + p[:, None] * 128 + tile * 16 + d[None, :],
        tl.dot(aa, scaled).to(tl.float16),
        mask=valid[:, None],
    )


def launch_wy16(q, k, v, decay, beta, cu, a16, u16, w16, g16, wk16, max_seqlen):
    """Launch independent U and W Tensor Core tiles for each BT16 block."""
    batches, chunks = cu.numel() - 1, triton.cdiv(max_seqlen, 16)
    if chunks:
        _wy_u16[(batches * chunks, 32, 8)](
            v,
            beta,
            cu,
            a16,
            u16,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _prefix_g16_grid[(batches * chunks, 32)](
            decay, cu, g16, CHUNKS=chunks, token_offset=0, num_warps=4, num_stages=1
        )
        _scale_w16[(batches * chunks, 32, 8)](
            k,
            beta,
            cu,
            g16,
            wk16,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _wy_w16[(batches * chunks, 32, 8)](
            wk16, a16, cu, w16, CHUNKS=chunks, token_offset=0, num_warps=4, num_stages=1
        )


def compute_wy16(q, k, v, decay, beta, cu, max_seqlen, a16):
    """Allocate and compute contiguous [B,C,32,16,128] U/W FP16 scratch."""
    import torch

    batches, chunks = cu.numel() - 1, triton.cdiv(max_seqlen, 16)
    shape = (batches, chunks, 32, 16, 128)
    u16 = torch.empty(shape, device=q.device, dtype=torch.float16)
    w16 = torch.empty_like(u16)
    g16 = torch.empty((batches, chunks, 32, 16), device=q.device, dtype=torch.float32)
    wk16 = torch.empty_like(u16)
    launch_wy16(q, k, v, decay, beta, cu, a16, u16, w16, g16, wk16, max_seqlen)
    return u16, w16


@triton.jit
def _residual16(
    u16,
    w16,
    cu,
    state,
    history16,
    r32,
    r16,
    chunk,
    CHUNKS: tl.constexpr,
    token_offset,
    RESET_FIRST: tl.constexpr = False,
):
    """Save H16 and form R32/R16; no R-dependent matrix product follows."""
    seq, h, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p, x, d = tl.arange(0, 16), vt * 16 + tl.arange(0, 16), tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    fresh = RESET_FIRST & (chunk + token_offset == 0)
    base = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    up = tl.load(
        u16 + base + p[:, None] * 128 + x[None, :], mask=valid[:, None], other=0.0
    )
    acc = tl.zeros((16, 16), tl.float32)
    for kt in range(0, 8):
        ww = tl.load(
            w16 + base + p[:, None] * 128 + kt * 16 + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        ss = tl.load(
            state + ((seq * 32 + h) * 128 + x[None, :]) * 128 + kt * 16 + d[:, None],
            mask=not fresh,
            other=0.0,
        ).to(tl.float16)
        acc += tl.dot(ww, ss)
    hp = (
        history16
        + (((seq * CHUNKS + chunk) * 32 + h) * 128 + x[:, None]) * 128
        + (tl.arange(0, 128))[None, :]
    )
    snap = tl.load(
        state
        + ((seq * 32 + h) * 128 + x[:, None]) * 128
        + (tl.arange(0, 128))[None, :],
        mask=not fresh,
        other=0.0,
    ).to(tl.float16)
    tl.store(hp, snap)
    value = up.to(tl.float32) - acc
    rp = r32 + base + p[:, None] * 128 + x[None, :]
    tl.store(rp, value, mask=valid[:, None])
    tl.store(
        r16 + base + p[:, None] * 128 + x[None, :],
        value.to(tl.float16),
        mask=valid[:, None],
    )


@triton.jit
def _rd_prepare16(r32, g16, cu, rd16, chunk, CHUNKS: tl.constexpr, token_offset):
    """Apply last-token decay then store [V,T] for the NN state MMA."""
    seq, h, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p, x = tl.arange(0, 16), vt * 16 + tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    gbase = ((seq * CHUNKS + chunk) * 32 + h) * 16
    g = tl.load(g16 + gbase + p, mask=valid, other=0.0)
    last = tl.sum(
        tl.where(valid & ((p == 15) | (p + 1 >= end - start)), g, 0.0), axis=0
    )
    base = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    tile_mask = (x[:, None] < 128) & valid[None, :]
    r = tl.load(r32 + base + p[None, :] * 128 + x[:, None], mask=tile_mask, other=0.0)
    # The transpose is materialized for a single 16x16x16 NN MMA in state.
    tl.store(
        rd16 + base + x[:, None] * 16 + p[None, :],
        (r * tl.exp(last - g)[None, :]).to(tl.float16),
        mask=tile_mask,
    )


@triton.jit
def _state_update16(
    k,
    g16,
    cu,
    rd16,
    state,
    chunk,
    CHUNKS: tl.constexpr,
    token_offset,
    RESET_FIRST: tl.constexpr = False,
):
    """FP32 persistent-state update from a pre-transposed RD16 tile."""
    seq, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    vt, kt = tile // 8, tile % 8
    p, x, d = tl.arange(0, 16), vt * 16 + tl.arange(0, 16), kt * 16 + tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    fresh = RESET_FIRST & (chunk + token_offset == 0)
    gbase = ((seq * CHUNKS + chunk) * 32 + h) * 16
    g = tl.load(g16 + gbase + p, mask=valid, other=0.0)
    last = tl.sum(
        tl.where(valid & ((p == 15) | (p + 1 >= end - start)), g, 0.0), axis=0
    )
    base = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    rd = tl.load(
        rd16 + base + x[:, None] * 16 + p[None, :], mask=valid[None, :], other=0.0
    )
    kk = tl.load(
        k + (t[:, None] * 16 + h // 2) * 128 + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    sp = state + ((seq * 32 + h) * 128 + x[:, None]) * 128 + d[None, :]
    old = tl.load(sp, mask=not fresh, other=0.0)
    value = tl.exp(last) * old + tl.dot(rd, kk).to(tl.float32)
    tl.store(sp, value, mask=(start < end) | fresh)


def compute_r_state16(k, decay, cu, max_seqlen, u16, w16):
    """Compute R/H snapshots and current-call FP32 final state for BT16 blocks."""
    import torch

    batches, chunks = cu.numel() - 1, triton.cdiv(max_seqlen, 16)
    history16 = torch.empty(
        (batches, chunks, 32, 128, 128), device=k.device, dtype=torch.float16
    )
    r32 = torch.empty(
        (batches, chunks, 32, 16, 128), device=k.device, dtype=torch.float32
    )
    r16 = torch.empty_like(u16)
    rd16 = torch.empty(
        (batches, chunks, 32, 128, 16), device=k.device, dtype=torch.float16
    )
    g16 = torch.empty((batches, chunks, 32, 16), device=k.device, dtype=torch.float32)
    state = torch.zeros((batches, 32, 128, 128), device=k.device, dtype=torch.float32)
    for chunk in range(chunks):
        _prefix_g16[(batches, 32)](
            decay,
            cu,
            g16,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _residual16[(batches, 32, 8)](
            u16,
            w16,
            cu,
            state,
            history16,
            r32,
            r16,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _rd_prepare16[(batches, 32, 8)](
            r32,
            g16,
            cu,
            rd16,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _state_update16[(batches, 32, 64)](
            k,
            g16,
            cu,
            rd16,
            state,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
    return history16, r32, r16, rd16, state


@triton.jit
def _qk16(q, k, cu, qk32, chunk, CHUNKS: tl.constexpr, token_offset):
    seq, h = tl.program_id(0), tl.program_id(1)
    p, d = tl.arange(0, 16), tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid, kh = start + p, (start < end) & (start + p < end), h // 2
    acc = tl.zeros((16, 16), tl.float32)
    for offset in range(0, 128, 16):
        qq = tl.load(
            q + (t[:, None] * 16 + kh) * 128 + offset + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kk = tl.load(
            k + (t[None, :] * 16 + kh) * 128 + offset + d[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        acc += tl.dot(qq, kk)
    base = ((seq * CHUNKS + chunk) * 32 + h) * 256 + p[:, None] * 16 + p[None, :]
    tl.store(qk32 + base, acc, mask=valid[:, None] & valid[None, :])


@triton.jit
def _coeff16(qk32, g16, cu, coeff16, chunk, CHUNKS: tl.constexpr, token_offset):
    seq, h = tl.program_id(0), tl.program_id(1)
    p = tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    base = ((seq * CHUNKS + chunk) * 32 + h) * 256 + p[:, None] * 16 + p[None, :]
    g = tl.load(g16 + ((seq * CHUNKS + chunk) * 32 + h) * 16 + p, mask=valid, other=0.0)
    causal = (p[:, None] >= p[None, :]) & valid[:, None] & valid[None, :]
    # Select the causal entries before exp; padded/future values never form 0*inf.
    value = tl.where(
        causal,
        tl.load(qk32 + base, mask=causal, other=0.0)
        * tl.exp(tl.where(causal, g[:, None] - g[None, :], 0.0)),
        0.0,
    )
    tl.store(coeff16 + base, value.to(tl.float16), mask=valid[:, None] & valid[None, :])


@triton.jit
def _prior16(q, cu, history16, prior32, chunk, CHUNKS: tl.constexpr, token_offset):
    seq, h, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p, x, d = tl.arange(0, 16), vt * 16 + tl.arange(0, 16), tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid, kh = start + p, (start < end) & (start + p < end), h // 2
    acc = tl.zeros((16, 16), tl.float32)
    hbase = ((seq * CHUNKS + chunk) * 32 + h) * 16384
    for offset in range(0, 128, 16):
        qq = tl.load(
            q + (t[:, None] * 16 + kh) * 128 + offset + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        hh = tl.load(history16 + hbase + x[None, :] * 128 + offset + d[:, None])
        acc += tl.dot(qq, hh)
    base = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    tl.store(prior32 + base + p[:, None] * 128 + x[None, :], acc, mask=valid[:, None])


@triton.jit
def _local16(coeff16, r16, cu, local32, chunk, CHUNKS: tl.constexpr, token_offset):
    seq, h, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p, x = tl.arange(0, 16), vt * 16 + tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    mbase = ((seq * CHUNKS + chunk) * 32 + h) * 256
    base = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    cc = tl.load(
        coeff16 + mbase + p[:, None] * 16 + p[None, :],
        mask=valid[:, None] & valid[None, :],
        other=0.0,
    )
    rr = tl.load(
        r16 + base + p[:, None] * 128 + x[None, :], mask=valid[:, None], other=0.0
    )
    tl.store(
        local32 + base + p[:, None] * 128 + x[None, :],
        tl.dot(cc, rr).to(tl.float32),
        mask=valid[:, None],
    )


@triton.jit
def _output16(
    prior32, local32, g16, cu, out, chunk, CHUNKS: tl.constexpr, token_offset
):
    seq, h, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p, x = tl.arange(0, 16), vt * 16 + tl.arange(0, 16)
    start, end = tl.load(cu + seq) + (chunk + token_offset) * 16, tl.load(cu + seq + 1)
    t, valid = start + p, (start < end) & (start + p < end)
    base = ((seq * CHUNKS + chunk) * 32 + h) * 2048
    g = tl.load(g16 + ((seq * CHUNKS + chunk) * 32 + h) * 16 + p, mask=valid, other=0.0)
    value = (
        tl.exp(g)[:, None]
        * tl.load(
            prior32 + base + p[:, None] * 128 + x[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        + tl.load(
            local32 + base + p[:, None] * 128 + x[None, :],
            mask=valid[:, None],
            other=0.0,
        )
    ) * (128**-0.5)
    tl.store(
        out + (t[:, None] * 32 + h) * 128 + x[None, :],
        value.to(tl.float16).to(tl.float32),
        mask=valid[:, None],
    )


def compute_output16(q, k, decay, cu, max_seqlen, history16, r16):
    """Return the checkpoint-rounded FP32 view of separated BT16 outputs."""
    import torch

    batches, chunks = cu.numel() - 1, triton.cdiv(max_seqlen, 16)
    qk32 = torch.empty(
        (batches, chunks, 32, 16, 16), device=q.device, dtype=torch.float32
    )
    coeff16 = torch.empty_like(qk32, dtype=torch.float16)
    prior32 = torch.empty(
        (batches, chunks, 32, 16, 128), device=q.device, dtype=torch.float32
    )
    local32 = torch.empty_like(prior32)
    g16 = torch.empty((batches, chunks, 32, 16), device=q.device, dtype=torch.float32)
    out = torch.empty((q.shape[0], 32, 128), device=q.device, dtype=torch.float32)
    for chunk in range(chunks):
        _prefix_g16[(batches, 32)](
            decay,
            cu,
            g16,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _qk16[(batches, 32)](
            q,
            k,
            cu,
            qk32,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _coeff16[(batches, 32)](
            qk32,
            g16,
            cu,
            coeff16,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _prior16[(batches, 32, 8)](
            q,
            cu,
            history16,
            prior32,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _local16[(batches, 32, 8)](
            coeff16,
            r16,
            cu,
            local32,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
        _output16[(batches, 32, 8)](
            prior32,
            local32,
            g16,
            cu,
            out,
            chunk=chunk,
            CHUNKS=chunks,
            token_offset=0,
            num_warps=4,
            num_stages=1,
        )
    return out


def stream_gdn16(q, k, v, decay, beta, cu, max_seqlen, *, out=None, state=None):
    """Run BT16 one token chunk at a time with O(B) workspace.

    ``token_offset`` selects the current sequence-local token chunk, while
    every scratch buffer uses the fixed storage chunk zero (``CHUNKS=1``).
    This distinction is what prevents ragged batches from allocating
    ``B * ceil(max_seqlen / 16)`` intermediate histories.

    Internal callers must provide ``max_seqlen > 0``.  The public
    :func:`chunk_gdn` entrypoint handles empty packed inputs before reaching
    this helper; a direct zero-length call does not initialize a newly
    allocated state buffer.
    """
    import torch

    batches = cu.numel() - 1
    # All temporaries are the current [B, 1, ...] chunk and are reused by
    # each ordered launch.  ``out`` may describe the whole packed input and
    # ``state`` one sequence tile, so its absolute token indices remain valid.
    matrix_shape = (batches, 1, 32, 16, 16)
    vector_shape = (batches, 1, 32, 16, 128)
    gram = torch.empty(matrix_shape, device=q.device, dtype=torch.float32)
    a16 = torch.empty(matrix_shape, device=q.device, dtype=torch.float16)
    u16 = torch.empty(vector_shape, device=q.device, dtype=torch.float16)
    w16 = torch.empty_like(u16)
    g16 = torch.empty((batches, 1, 32, 16), device=q.device, dtype=torch.float32)
    wk16 = torch.empty_like(u16)
    history16 = torch.empty(
        (batches, 1, 32, 128, 128), device=q.device, dtype=torch.float16
    )
    r32 = torch.empty(vector_shape, device=q.device, dtype=torch.float32)
    r16 = torch.empty_like(u16)
    rd16 = torch.empty((batches, 1, 32, 128, 16), device=q.device, dtype=torch.float16)
    qk32 = torch.empty(matrix_shape, device=q.device, dtype=torch.float32)
    coeff16 = torch.empty_like(qk32, dtype=torch.float16)
    prior32 = torch.empty(vector_shape, device=q.device, dtype=torch.float32)
    local32 = torch.empty_like(prior32)
    if state is None:
        state = torch.empty(
            (batches, 32, 128, 128), device=q.device, dtype=torch.float32
        )
    if out is None:
        out = torch.empty((q.shape[0], 32, 128), device=q.device, dtype=torch.float32)

    for token_chunk in range(triton.cdiv(max_seqlen, 16)):
        # Storage chunk is always 0; token_offset advances the packed input.
        _kkt16[(batches, 1, 32)](
            q,
            k,
            cu,
            gram,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _inverse16_col_f32[(batches, 1, 512)](
            gram,
            decay,
            beta,
            cu,
            a16,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _wy_u16[(batches, 32, 8)](
            v,
            beta,
            cu,
            a16,
            u16,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _prefix_g16_grid[(batches, 32)](
            decay,
            cu,
            g16,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _scale_w16[(batches, 32, 8)](
            k,
            beta,
            cu,
            g16,
            wk16,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _wy_w16[(batches, 32, 8)](
            wk16,
            a16,
            cu,
            w16,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )

        # _residual16 snapshots old state before _state_update16 mutates it.
        _residual16[(batches, 32, 8)](
            u16,
            w16,
            cu,
            state,
            history16,
            r32,
            r16,
            chunk=0,
            CHUNKS=1,
            token_offset=token_chunk,
            RESET_FIRST=True,
            num_warps=4,
            num_stages=1,
        )
        _rd_prepare16[(batches, 32, 8)](
            r32,
            g16,
            cu,
            rd16,
            chunk=0,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _state_update16[(batches, 32, 64)](
            k,
            g16,
            cu,
            rd16,
            state,
            chunk=0,
            CHUNKS=1,
            token_offset=token_chunk,
            RESET_FIRST=True,
            num_warps=4,
            num_stages=1,
        )

        _qk16[(batches, 32)](
            q,
            k,
            cu,
            qk32,
            chunk=0,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _coeff16[(batches, 32)](
            qk32,
            g16,
            cu,
            coeff16,
            chunk=0,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _prior16[(batches, 32, 8)](
            q,
            cu,
            history16,
            prior32,
            chunk=0,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _local16[(batches, 32, 8)](
            coeff16,
            r16,
            cu,
            local32,
            chunk=0,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
        _output16[(batches, 32, 8)](
            prior32,
            local32,
            g16,
            cu,
            out,
            chunk=0,
            CHUNKS=1,
            token_offset=token_chunk,
            num_warps=4,
            num_stages=1,
        )
    return out, state
