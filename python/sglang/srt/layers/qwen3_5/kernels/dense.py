"""Conservative SM70 Triton dense kernels.

Volta lowering in Triton 2.3.1 is sensitive to layouts and pipeline stages.
These kernels intentionally use the small configurations covered by the
compatibility tests: 32-cube NT GEMM and one pipeline stage.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _nt_gemm_kernel(
    x, w, out, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    sxm: tl.constexpr, sxk: tl.constexpr, swn: tl.constexpr, swk: tl.constexpr,
    som: tl.constexpr, son: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, k, BLOCK_K):
        k_offsets = start_k + offs_k
        a = tl.load(
            x + offs_m[:, None] * sxm + k_offsets[None, :] * sxk,
            mask=(offs_m[:, None] < m) & (k_offsets[None, :] < k), other=0.0,
        )
        b = tl.load(
            w + offs_n[None, :] * swn + k_offsets[:, None] * swk,
            mask=(offs_n[None, :] < n) & (k_offsets[:, None] < k), other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(out + offs_m[:, None] * som + offs_n[None, :] * son, acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def _embedding_kernel(ids, table, out, rows: tl.constexpr, hidden: tl.constexpr,
                      sid: tl.constexpr, str_: tl.constexpr, stc: tl.constexpr,
                      som: tl.constexpr, son: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    token = tl.load(ids + row * sid)
    valid_token = (token >= 0) & (token < rows)
    values = tl.load(table + token * str_ + cols * stc, mask=valid_token & (cols < hidden), other=0.0)
    tl.store(out + row * som + cols * son, values, mask=cols < hidden)


def nt_gemm(x, weight, out=None):
    """Compute ``x @ weight.T`` with FP32 accumulation and FP16 output."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("nt_gemm expects rank-2 tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("incompatible NT GEMM shapes")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("nt_gemm is a CUDA Triton operation")
    if x.stride(-1) != 1 or weight.stride(-1) != 1:
        raise ValueError("nt_gemm only accepts inner-contiguous operands on SM70")
    if x.dtype != weight.dtype or x.dtype != torch.float16:
        raise TypeError("x and weight must have the same dtype")
    m, k = x.shape
    n = weight.shape[0]
    if out is None:
        out = x.new_empty((m, n))
    if (out.shape != (m, n) or out.dtype != x.dtype or not out.is_cuda or
            out.device != x.device or not out.is_contiguous()):
        raise ValueError("invalid NT GEMM output")
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    _nt_gemm_kernel[grid](
        x, weight, out, m, n, k,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        num_warps=4, num_stages=1,
    )
    return out


def embedding(ids, table, out=None, validate_ids=False):
    if ids.ndim != 1 or table.ndim != 2 or not ids.is_cuda or not table.is_cuda:
        raise ValueError("embedding expects CUDA ids [T] and table [V,H]")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("ids must be an integer tensor")
    if validate_ids and not bool(torch.all((ids >= 0) & (ids < table.shape[0]))):
        raise IndexError("embedding token id is outside the table")
    tokens, hidden = ids.numel(), table.shape[1]
    if hidden > 8192:
        raise ValueError("embedding hidden size exceeds supported block")
    if out is None:
        out = table.new_empty((tokens, hidden))
    if out.shape != (tokens, hidden) or out.dtype != table.dtype or out.device != table.device:
        raise ValueError("invalid embedding output")
    block = triton.next_power_of_2(hidden)
    _embedding_kernel[(tokens,)](
        ids, table, out, table.shape[0], hidden,
        ids.stride(0), table.stride(0), table.stride(1), out.stride(0), out.stride(1),
        BLOCK=block, num_warps=4,
    )
    return out
