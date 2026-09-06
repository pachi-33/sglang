"""Elementwise and normalization Triton kernels used by Qwen3.5."""

import triton
import triton.language as tl


@triton.jit
def _gemma_rms_kernel(x, weight, out, rows: tl.constexpr, hidden: tl.constexpr,
                      sxm: tl.constexpr, sxn: tl.constexpr, sw: tl.constexpr,
                      som: tl.constexpr, son: tl.constexpr, eps: tl.constexpr,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    value = tl.load(x + row * sxm + offs * sxn, mask=offs < hidden, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(value * value, axis=0) / hidden + eps)
    scale = tl.load(weight + offs * sw, mask=offs < hidden, other=0.0).to(tl.float32)
    result = value * inv * (1.0 + scale)
    tl.store(out + row * som + offs * son, result, mask=offs < hidden)


@triton.jit
def _gated_rms_kernel(x, gate, weight, out, rows: tl.constexpr, hidden: tl.constexpr,
                      sxm: tl.constexpr, sxn: tl.constexpr, sgm: tl.constexpr, sgn: tl.constexpr,
                      sw: tl.constexpr, som: tl.constexpr, son: tl.constexpr,
                      eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    value = tl.load(x + row * sxm + offs * sxn, mask=offs < hidden, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(value * value, axis=0) / hidden + eps)
    z = tl.load(gate + row * sgm + offs * sgn, mask=offs < hidden, other=0.0).to(tl.float32)
    scale = tl.load(weight + offs * sw, mask=offs < hidden, other=0.0).to(tl.float32)
    result = value * inv * scale * z / (1.0 + tl.exp(-z))
    tl.store(out + row * som + offs * son, result, mask=offs < hidden)


@triton.jit
def _binary_kernel(x, y, out, count: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x + offs, mask=offs < count)
    b = tl.load(y + offs, mask=offs < count)
    if OP == 0:
        result = a + b
    elif OP == 1:
        b32 = b.to(tl.float32)
        result = a.to(tl.float32) * (1.0 / (1.0 + tl.exp(-b32)))
    else:
        a32 = a.to(tl.float32)
        result = a32 / (1.0 + tl.exp(-a32))
    tl.store(out + offs, result, mask=offs < count)


def _check_matrix(x, name):
    if x.ndim != 2 or not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA rank-2 tensor")
    if x.shape[1] > 8192:
        raise ValueError("hidden dimension exceeds supported normalization block")


def gemma_rms_norm(x, weight, eps=1e-6, out=None):
    _check_matrix(x, "x")
    if weight.ndim != 1 or weight.numel() != x.shape[1] or not weight.is_cuda:
        raise ValueError("weight must be CUDA [hidden]")
    if out is None:
        out = x.new_empty(x.shape)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("invalid Gemma RMSNorm output")
    block = triton.next_power_of_2(x.shape[1])
    _gemma_rms_kernel[(x.shape[0],)](
        x, weight, out, x.shape[0], x.shape[1], x.stride(0), x.stride(1), weight.stride(0),
        out.stride(0), out.stride(1), eps, BLOCK=block, num_warps=4,
    )
    return out


def gated_rms_norm_silu(x, gate, weight, eps=1e-6, out=None):
    _check_matrix(x, "x")
    if gate.shape != x.shape or weight.ndim != 1 or weight.numel() != x.shape[1]:
        raise ValueError("gated RMSNorm shape mismatch")
    if not gate.is_cuda or not weight.is_cuda:
        raise ValueError("gated RMSNorm expects CUDA tensors")
    if out is None:
        out = x.new_empty(x.shape)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("invalid gated RMSNorm output")
    block = triton.next_power_of_2(x.shape[1])
    _gated_rms_kernel[(x.shape[0],)](
        x, gate, weight, out, x.shape[0], x.shape[1], x.stride(0), x.stride(1),
        gate.stride(0), gate.stride(1), weight.stride(0), out.stride(0), out.stride(1), eps,
        BLOCK=block, num_warps=4,
    )
    return out


def residual_add(x, residual, out=None):
    if x.shape != residual.shape or not x.is_cuda or not residual.is_cuda or not x.is_contiguous() or not residual.is_contiguous():
        raise ValueError("residual tensors must be matching CUDA tensors")
    if out is None:
        out = x.new_empty(x.shape)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError("residual output must be contiguous and match x")
    _binary_kernel[(triton.cdiv(x.numel(), 256),)](x, residual, out, x.numel(), OP=0, BLOCK=256, num_warps=4)
    return out


def sigmoid_mul(x, gate, out=None):
    if x.shape != gate.shape or not x.is_cuda or not gate.is_cuda or not x.is_contiguous() or not gate.is_contiguous():
        raise ValueError("sigmoid_mul tensors must be matching CUDA tensors")
    if out is None:
        out = x.new_empty(x.shape)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError("sigmoid_mul output must be contiguous and match x")
    _binary_kernel[(triton.cdiv(x.numel(), 256),)](x, gate, out, x.numel(), OP=1, BLOCK=256, num_warps=4)
    return out


def silu(x, out=None):
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("silu expects CUDA tensor")
    if out is None:
        out = x.new_empty(x.shape)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError("silu output must be contiguous and match x")
    _binary_kernel[(triton.cdiv(x.numel(), 256),)](x, x, out, x.numel(), OP=2, BLOCK=256, num_warps=4)
    return out
