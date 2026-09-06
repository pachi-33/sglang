"""SM70-safe Tensor-Core causal GQA using K=512 online slabs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk(
    q,
    k,
    cu,
    scores,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    maxs: tl.constexpr,
    batch: tl.constexpr,
    slab: tl.constexpr,
    nblocks: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    qb = tl.program_id(0)
    h = tl.program_id(1)
    bz = tl.program_id(2)
    b = bz // nblocks
    nb = bz % nblocks
    start = tl.load(cu + b)
    end = tl.load(cu + b + 1)
    length = end - start
    rm = qb * BM + tl.arange(0, BM)
    rn = slab + nb * BN + tl.arange(0, BN)
    tok = start + rm
    ktok = start + rn
    kh = h // (hq // hkv)
    acc = tl.zeros((BM, BN), tl.float32)
    for dk in range(0, 256, BK):
        d = dk + tl.arange(0, BK)
        a = tl.load(
            q + tok[:, None] * hq * 256 + h * 256 + d[None, :],
            mask=rm[:, None] < length,
            other=0.0,
        )
        # k is [T,Hkv,D], loaded transposed for TC QK
        bb = tl.load(
            k + ktok[None, :] * hkv * 256 + kh * 256 + d[:, None],
            mask=rn[None, :] < length,
            other=0.0,
        )
        acc += tl.dot(a, bb)
    valid = (
        (rm[:, None] < length) & (rn[None, :] < length) & (rn[None, :] <= rm[:, None])
    )
    tl.store(
        scores + tok[:, None] * hq * 512 + h * 512 + (rn[None, :] - slab),
        acc * 0.0625,
        mask=valid,
    )


@triton.jit
def _softmax(
    scores,
    p,
    local_m,
    local_l,
    cu,
    hq: tl.constexpr,
    batch: tl.constexpr,
    slab: tl.constexpr,
    C: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    start = 0
    end = 0
    for b in range(0, batch):
        # API admits int64 cu_seqlens, but this implementation caps T at
        # 2048, so use one loop-carried int32 type for either input dtype.
        s = tl.load(cu + b).to(tl.int32)
        e = tl.load(cu + b + 1).to(tl.int32)
        hit = (t >= s) & (t < e)
        start = tl.where(hit, s, start)
        end = tl.where(hit, e, end)
    row = t - start
    valid_row = t < end
    m = -float("inf")
    for off in range(0, C, 32):
        n = off + tl.arange(0, 32)
        valid = valid_row & (n + slab < end - start) & (n + slab <= row)
        x = tl.load(scores + t * hq * C + h * C + n, mask=valid, other=-float("inf"))
        m = tl.maximum(m, tl.max(x, axis=0))
    m = tl.where(valid_row, m, -float("inf"))
    l = 0.0
    for off in range(0, C, 32):
        n = off + tl.arange(0, 32)
        valid = valid_row & (n + slab < end - start) & (n + slab <= row)
        x = tl.load(scores + t * hq * C + h * C + n, mask=valid, other=-float("inf"))
        prob = tl.where(valid, tl.exp(x - m), 0.0)
        l += tl.sum(prob, axis=0)
        tl.store(p + t * hq * C + h * C + n, prob, mask=n < C)
    tl.store(local_m + t * hq + h, m)
    tl.store(local_l + t * hq + h, l)


@triton.jit
def _transpose_v(v, vt, tokens: tl.constexpr, hkv: tl.constexpr, SV0: tl.constexpr):
    t = tl.program_id(0)
    hd = tl.program_id(1)
    h = hd // 16
    d = hd % 16 * 16 + tl.arange(0, 16)
    tl.store(
        vt + h * 256 * tokens + d * tokens + t,
        tl.load(v + t * SV0 + h * 256 + d, mask=d < 256),
        mask=d < 256,
    )


@triton.jit
def _pv(
    p,
    vt,
    cu,
    partial,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    tokens: tl.constexpr,
    batch: tl.constexpr,
    slab: tl.constexpr,
    C: tl.constexpr,
):
    qb = tl.program_id(0)
    h = tl.program_id(1)
    bz = tl.program_id(2)
    b = bz // 16
    dt = bz % 16
    start = tl.load(cu + b)
    end = tl.load(cu + b + 1)
    length = end - start
    rm = qb * 16 + tl.arange(0, 16)
    t = start + rm
    kh = h // (hq // hkv)
    d = dt * 16 + tl.arange(0, 16)
    acc = tl.zeros((16, 16), tl.float32)
    for off in range(0, C, 16):
        n = off + tl.arange(0, 16)
        key = start + slab + n
        pp = tl.load(
            p + t[:, None] * hq * C + h * C + n[None, :],
            mask=(rm[:, None] < length) & (n[None, :] < C),
            other=0.0,
        ).to(tl.float16)
        vv = tl.load(
            vt + kh * 256 * tokens + d[None, :] * tokens + key[:, None],
            mask=(key[:, None] < end) & (d[None, :] < 256),
            other=0.0,
        )
        acc += tl.dot(pp, vv)
    tl.store(
        partial + t[:, None] * hq * 256 + h * 256 + d[None, :],
        acc,
        mask=(rm[:, None] < length) & (d[None, :] < 256),
    )


@triton.jit
def _merge(
    partial,
    lm,
    ll,
    m,
    l,
    acc,
    hq: tl.constexpr,
    FIRST: tl.constexpr,
    FINAL: tl.constexpr,
    out,
    BLOCK: tl.constexpr,
):
    # One CTA owns the complete (token, query-head) row.  In particular, m
    # and l are scalar online-softmax state: splitting D across CTAs races
    # those stores (and applies a slab more than once).
    i = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    oldm = -float("inf") if FIRST else tl.load(m + i * hq + h)
    oldl = 0.0 if FIRST else tl.load(l + i * hq + h)
    nm = tl.load(lm + i * hq + h)
    nl = tl.load(ll + i * hq + h)
    newm = tl.maximum(oldm, nm)
    a = tl.where(oldl > 0, tl.exp(oldm - newm), 0.0)
    b = tl.where(nl > 0, tl.exp(nm - newm), 0.0)
    newl = oldl * a + nl * b
    old = (
        tl.zeros((BLOCK,), tl.float32)
        if FIRST
        else tl.load(acc + i * hq * 256 + h * 256 + d, mask=d < 256, other=0.0)
    )
    part = tl.load(partial + i * hq * 256 + h * 256 + d, mask=d < 256, other=0.0)
    val = old * a + part * b
    tl.store(acc + i * hq * 256 + h * 256 + d, val, mask=d < 256)
    tl.store(m + i * hq + h, newm)
    tl.store(l + i * hq + h, newl)
    if FINAL:
        tl.store(out + i * hq * 256 + h * 256 + d, val / newl, mask=d < 256)


def causal_gqa_slab(q, k, v, cu, max_seqlen, out=None):
    if q.numel() == 0:
        return q.new_empty(q.shape)
    tokens, hq, _ = q.shape
    batch = cu.numel() - 1
    columns = 512
    vt = torch.empty((2, 256, tokens), device=q.device, dtype=q.dtype)
    scores = torch.empty((tokens, hq, columns), device=q.device, dtype=torch.float32)
    p = torch.empty_like(scores, dtype=torch.float16)
    local_m = torch.empty((tokens, hq), device=q.device, dtype=torch.float32)
    local_l = torch.empty_like(local_m)
    m = torch.empty_like(local_m)
    l = torch.empty_like(local_m)
    acc = torch.empty((tokens, hq, 256), device=q.device, dtype=torch.float32)
    partial = torch.empty_like(acc)
    if out is None:
        out = torch.empty_like(q)
    _transpose_v[(tokens, 32)](v, vt, tokens, 2, v.stride(0), num_warps=4)
    for slab in range(0, max_seqlen, columns):
        nblocks = triton.cdiv(min(columns, max_seqlen - slab), 16)
        _qk[(triton.cdiv(max_seqlen, 16), hq, batch * nblocks)](
            q,
            k,
            cu,
            scores,
            hq,
            2,
            max_seqlen,
            batch,
            slab,
            nblocks,
            BM=16,
            BN=16,
            BK=16,
            num_warps=4,
            num_stages=1,
        )
        _softmax[(tokens, hq)](
            scores, p, local_m, local_l, cu, hq, batch, slab, columns, num_warps=4
        )
        _pv[(triton.cdiv(max_seqlen, 16), hq, batch * 16)](
            p,
            vt,
            cu,
            partial,
            hq,
            2,
            tokens,
            batch,
            slab,
            columns,
            num_warps=4,
            num_stages=1,
        )
        _merge[(tokens, hq)](
            partial,
            local_m,
            local_l,
            m,
            l,
            acc,
            hq,
            slab == 0,
            slab + columns >= max_seqlen,
            out,
            BLOCK=256,
            num_warps=4,
        )
    return out
