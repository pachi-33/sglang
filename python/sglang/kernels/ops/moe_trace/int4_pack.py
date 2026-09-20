"""Triton implementation of groupwise int4 activation packing.

This module deliberately keeps Triton optional.  The trace codec imports on
CPU-only installations, while attempting to use this function for a CUDA
tensor without Triton gives a useful error.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:  # Triton is intentionally an optional dependency for CPU-only users.
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed build
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _int4_quantize_pack_kernel(
        x,
        scales,
        packed,
        x_row_stride,
        x_col_stride,
        scales_row_stride,
        scales_col_stride,
        packed_row_stride,
        packed_col_stride,
        H: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        group = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_SIZE)
        columns = group * GROUP_SIZE + offsets
        valid = (offsets < GROUP_SIZE) & (columns < H)
        values = tl.load(
            x + row * x_row_stride + columns * x_col_stride,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=0)
        scale = tl.where(amax == 0.0, 1.0, amax / 7.0)
        tl.store(scales + row * scales_row_stride + group * scales_col_stride, scale)

        # GROUP_SIZE is even, so every packed byte belongs to precisely one
        # group.  Keeping reduction and quantization in this program preserves
        # the full-precision max(abs(group)) / 7 scale for q, even though the
        # externally stored scale is fp16.
        pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
        col0 = group * GROUP_SIZE + pair_offsets * 2
        col1 = col0 + 1
        valid0 = (pair_offsets * 2 < GROUP_SIZE) & (col0 < H)
        valid1 = (pair_offsets * 2 + 1 < GROUP_SIZE) & (col1 < H)

        x0 = tl.load(
            x + row * x_row_stride + col0 * x_col_stride,
            mask=valid0,
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            x + row * x_row_stride + col1 * x_col_stride,
            mask=valid1,
            other=0.0,
        ).to(tl.float32)
        q0 = libdevice.rint(x0 / scale).to(tl.int32)
        q1 = libdevice.rint(x1 / scale).to(tl.int32)
        q0 = tl.maximum(-7, tl.minimum(7, q0))
        q1 = tl.maximum(-7, tl.minimum(7, q1))
        packed_byte = (q0 & 0xF) | ((q1 & 0xF) << 4)
        byte_offsets = group * (GROUP_SIZE // 2) + pair_offsets
        valid_byte = valid0
        tl.store(
            packed + row * packed_row_stride + byte_offsets * packed_col_stride,
            packed_byte,
            mask=valid_byte,
        )


def quantize_and_pack_int4_cuda(
    x: torch.Tensor,
    group_size: int,
    out_q: torch.Tensor,
    out_scales: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize CUDA ``x`` into caller-owned packed int4 and fp16 buffers.

    All tensor dimensions used for launch geometry are Python shape metadata;
    this function never reads a device scalar or synchronizes the host.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required to quantize CUDA MoE trace tensors")
    if not x.is_cuda:
        raise ValueError("quantize_and_pack_int4_cuda requires a CUDA tensor")
    if x.ndim != 2:
        raise ValueError(
            f"expected x to have shape [rows, hidden], got {tuple(x.shape)}"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported input dtype: {x.dtype}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if group_size % 2:
        raise ValueError("CUDA int4 packing requires an even group_size")

    rows, hidden = x.shape
    packed_width = (hidden + 1) // 2
    num_groups = (hidden + group_size - 1) // group_size
    if out_q.shape != (rows, packed_width) or out_q.dtype != torch.uint8:
        raise ValueError(
            "out_q must be uint8 with shape "
            f"({rows}, {packed_width}), got {out_q.dtype} {tuple(out_q.shape)}"
        )
    if out_scales.shape != (rows, num_groups) or out_scales.dtype != torch.float16:
        raise ValueError(
            "out_scales must be float16 with shape "
            f"({rows}, {num_groups}), got {out_scales.dtype} {tuple(out_scales.shape)}"
        )
    if out_q.device != x.device or out_scales.device != x.device:
        raise ValueError("x, out_q, and out_scales must be on the same device")
    if rows == 0 or hidden == 0:
        return out_q, out_scales

    scale_block = triton.next_power_of_2(group_size)
    _int4_quantize_pack_kernel[(rows, num_groups)](
        x,
        out_scales,
        out_q,
        x.stride(0),
        x.stride(1),
        out_scales.stride(0),
        out_scales.stride(1),
        out_q.stride(0),
        out_q.stride(1),
        H=hidden,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=scale_block,
        num_warps=4,
    )
    return out_q, out_scales
