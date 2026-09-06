"""Stateless Gated DeltaNet primitives for SM70."""

import triton
import triton.language as tl
import torch

from .gdn_chunk import stream_gdn16



@triton.jit
def _conv4_silu(x, weight, bias, cu, out, channels: tl.constexpr, batch: tl.constexpr,
                BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr):
    token = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # cu_seqlens is intentionally a trusted packed-sequence descriptor.  A
    # short scan is cheaper than materialising a token-to-sequence map.
    start = tl.zeros((), tl.int64)
    for b in range(0, batch):
        s = tl.load(cu + b)
        e = tl.load(cu + b + 1)
        start = tl.where((token >= s) & (token < e), s, start)
    if HAS_BIAS:
        value = tl.load(bias + channel, mask=channel < channels, other=0.0).to(tl.float32)
    else:
        value = tl.zeros((BLOCK,), tl.float32)
    for tap in range(0, 4):
        index = token - 3 + tap
        valid = index >= start
        value += tl.load(x + index * channels + channel, mask=valid & (channel < channels), other=0.0).to(tl.float32) * tl.load(weight + channel * 4 + tap, mask=channel < channels, other=0.0).to(tl.float32)
    tl.store(out + token * channels + channel, value / (1.0 + tl.exp(-value)), mask=channel < channels)


@triton.jit
def _l2_qk(x, out, heads: tl.constexpr, dim: tl.constexpr, tokens: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    value = tl.load(x + (token * heads + head) * dim + d, mask=d < dim, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(value * value, axis=0) + 1e-6)
    tl.store(out + (token * heads + head) * dim + d, value * inv, mask=d < dim)


@triton.jit
def _prepare_gates(a, b, a_log, dt_bias, decay, beta, heads: tl.constexpr,
                   BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.arange(0, BLOCK)
    mask = head < heads
    av = tl.load(a + token * heads + head, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(b + token * heads + head, mask=mask, other=0.0).to(tl.float32)
    x = av + tl.load(dt_bias + head, mask=mask, other=0.0).to(tl.float32)
    # log1p(exp(x)) written this way remains finite for checkpoint outliers.
    softplus = tl.maximum(x, 0.0) + tl.math.log1p(tl.exp(-tl.abs(x)))
    g = -tl.exp(tl.load(a_log + head, mask=mask, other=0.0).to(tl.float32)) * softplus
    # Checkpoint boundary: beta is explicitly rounded to FP16 before recurrence.
    beta16 = (1.0 / (1.0 + tl.exp(-bv))).to(tl.float16)
    tl.store(decay + token * heads + head, g, mask=mask)
    tl.store(beta + token * heads + head, beta16, mask=mask)


@triton.jit
def _recurrent(q, k, v, decay, beta, cu, out, state, hq: tl.constexpr, hv: tl.constexpr,
               dim: tl.constexpr, max_seqlen: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row = tl.program_id(2)
    start = tl.load(cu + b)
    end = tl.load(cu + b + 1)
    kh = h // (hv // hq)
    d = tl.arange(0, BLOCK)
    s = tl.zeros((BLOCK,), tl.float32)
    for local_t in range(0, max_seqlen):
        token = start + local_t
        valid = token < end
        kv = tl.load(k + (token * hq + kh) * dim + d, mask=valid & (d < dim), other=0.0).to(tl.float32)
        qv = tl.load(q + (token * hq + kh) * dim + d, mask=valid & (d < dim), other=0.0).to(tl.float32)
        g = tl.load(decay + token * hv + h, mask=valid, other=0.0)
        bt = tl.load(beta + token * hv + h, mask=valid, other=0.0).to(tl.float32)
        decayed = s * tl.exp(g)
        prediction = tl.sum(decayed * kv, axis=0)
        vv = tl.load(v + (token * hv + h) * dim + row, mask=valid, other=0.0).to(tl.float32)
        r = bt * (vv - prediction)
        s = decayed + r * kv
        result = tl.sum(s * qv, axis=0) * scale
        tl.store(out + (token * hv + h) * dim + row, result, mask=valid)
    tl.store(state + ((b * hv + h) * dim + row) * dim + d, s, mask=d < dim)


def _check_cuda(tensor, name, dtype=None):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_cu(cu_seqlens, device):
    _check_cuda(cu_seqlens, "cu_seqlens")
    if cu_seqlens.device != device or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a same-device rank-1 tensor with at least two entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must have int32 or int64 dtype")


def depthwise_conv4_silu(x, weight, bias, cu_seqlens):
    """Packed causal depthwise Conv4 followed by SiLU, with sequence resets."""
    _check_cuda(x, "x", torch.float16)
    _check_cuda(weight, "weight", torch.float16)
    if bias is not None:
        _check_cuda(bias, "bias", torch.float16)
    _check_cu(cu_seqlens, x.device)
    if x.ndim != 2 or weight.shape != (x.shape[1], 4) or weight.device != x.device or (bias is not None and (bias.shape != (x.shape[1],) or bias.device != x.device)):
        raise ValueError("expected x[T,C], weight[C,4], and optional bias[C]")
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    block = 128
    # x is a safe, already-allocated placeholder when HAS_BIAS is false; the
    # constexpr branch emits no load from that argument.
    _conv4_silu[(x.shape[0], triton.cdiv(x.shape[1], block))](x, weight, bias if bias is not None else x, cu_seqlens, out, x.shape[1], cu_seqlens.numel() - 1, BLOCK=block, HAS_BIAS=bias is not None, num_warps=4, num_stages=1)
    return out


def l2_normalize_qk(q, k):
    _check_cuda(q, "q", torch.float16)
    _check_cuda(k, "k", torch.float16)
    if q.ndim != 3 or k.ndim != 3 or q.shape[-1] != 128 or k.shape[-1] != 128 or q.shape[:2] != k.shape[:2] or q.device != k.device:
        raise ValueError("q/k must have matching [T,H,128] shapes")
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    _l2_qk[(q.shape[0], q.shape[1])](q, q_out, q.shape[1], 128, q.shape[0], BLOCK=128, num_warps=4)
    _l2_qk[(k.shape[0], k.shape[1])](k, k_out, k.shape[1], 128, k.shape[0], BLOCK=128, num_warps=4)
    return q_out, k_out


def prepare_gates(a, b, a_log, dt_bias):
    for tensor, name in ((a, "a"), (b, "b"), (a_log, "a_log"), (dt_bias, "dt_bias")):
        _check_cuda(tensor, name, torch.float16)
    if a.ndim != 2 or not 1 <= a.shape[1] <= 32 or b.shape != a.shape or a_log.shape != (a.shape[1],) or dt_bias.shape != (a.shape[1],) or b.device != a.device or a_log.device != a.device or dt_bias.device != a.device:
        raise ValueError("expected a/b[T,H] and a_log/dt_bias[H] on one device")
    decay = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(a, dtype=torch.float16)
    if a.numel() != 0:
        _prepare_gates[(a.shape[0],)](a, b, a_log, dt_bias, decay, beta, a.shape[1], BLOCK=32, num_warps=4, num_stages=1)
    return decay, beta


def recurrent_gdn(q, k, v, decay, beta, cu_seqlens, max_seqlen):
    """Return FP32 output and final FP32 [B,32,128,128] state; no cache."""
    _validate_gdn_inputs(q, k, v, decay, beta, cu_seqlens, max_seqlen)
    tokens, hq, dim = q.shape
    hv = v.shape[1]
    out = torch.empty((tokens, hv, dim), device=q.device, dtype=torch.float32)
    state = torch.empty((cu_seqlens.numel() - 1, hv, dim, dim), device=q.device, dtype=torch.float32)
    if tokens:
        _recurrent[(cu_seqlens.numel() - 1, hv, dim)](q, k, v, decay, beta, cu_seqlens, out, state, hq, hv, dim, max_seqlen, dim ** -0.5, BLOCK=128, num_warps=4, num_stages=1)
    else:
        state.zero_()
    return out, state


def _validate_gdn_inputs(q, k, v, decay, beta, cu_seqlens, max_seqlen):
    """Validate tensor metadata without synchronizing to inspect packed cu.

    Callers provide trusted ``cu_seqlens`` contents: it starts at zero, ends
    at T, is nondecreasing, each sequence length is at most ``max_seqlen``,
    and its positions describe each packed token exactly once.  Kernels read
    those values on device; this launcher deliberately performs no host scan.
    """
    for tensor, name, dtype in ((q, "q", torch.float16), (k, "k", torch.float16),
                                (v, "v", torch.float16), (decay, "decay", torch.float32),
                                (beta, "beta", torch.float16)):
        _check_cuda(tensor, name, dtype)
    _check_cu(cu_seqlens, q.device)
    if isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int) or not 0 <= max_seqlen <= 2048:
        raise ValueError("max_seqlen must be a non-bool Python int in [0, 2048]")
    if (q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or q.shape[-1] != 128 or
            q.shape != k.shape or v.shape != (q.shape[0], 32, 128) or
            decay.shape != beta.shape or decay.shape != (q.shape[0], 32) or
            q.shape[1] != 16 or any(t.device != q.device for t in (k, v, decay, beta))):
        raise ValueError("expected q/k[T,16,128], v[T,32,128], decay/beta[T,32]")
    if q.shape[0] > 2048:
        raise ValueError("packed token count must be at most 2048")
    if q.shape[0] and max_seqlen == 0:
        raise ValueError("max_seqlen must be positive for nonempty packed inputs")


def chunk_gdn(q, k, v, decay, beta, cu_seqlens, max_seqlen, chunk_size=16):
    """BT16 Gated DeltaNet recurrence using the chunk WY form.

    This public entrypoint is reserved for the separated WY stages.  It is not
    allowed to silently fall back to the diagnostic scalar recurrence.
    """
    _validate_gdn_inputs(q, k, v, decay, beta, cu_seqlens, max_seqlen)
    if chunk_size != 16:
        raise ValueError("only the SM70-safe BT16 chunk size is supported")
    tokens, dim = q.shape[0], q.shape[-1]
    batches = cu_seqlens.numel() - 1
    if tokens == 0:
        return (torch.empty((0, 32, dim), device=q.device, dtype=torch.float32),
                torch.zeros((batches, 32, dim, dim), device=q.device, dtype=torch.float32))
    return stream_gdn16(q, k, v, decay, beta, cu_seqlens, max_seqlen)
