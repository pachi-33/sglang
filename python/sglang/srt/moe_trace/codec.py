"""Groupwise int4 codec for optional MoE router-input traces."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_input(x: torch.Tensor, group_size: int) -> Tuple[int, int, int, int]:
    if x.ndim != 2:
        raise ValueError(
            f"expected x to have shape [rows, hidden], got {tuple(x.shape)}"
        )
    if x.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise TypeError(f"unsupported input dtype: {x.dtype}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if group_size % 2:
        raise ValueError("group_size must be even for packed int4 quantization")
    rows, hidden = x.shape
    return rows, hidden, (hidden + 1) // 2, (hidden + group_size - 1) // group_size


def _validate_outputs(
    x: torch.Tensor,
    out_q: torch.Tensor,
    out_scales: torch.Tensor,
    packed_width: int,
    num_groups: int,
) -> None:
    rows = x.shape[0]
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


def pack_int4(q: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pack signed int4 values with the even value in each byte's low nibble."""
    if q.ndim != 2:
        raise ValueError(
            f"expected q to have shape [rows, hidden], got {tuple(q.shape)}"
        )
    if q.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"q must have an integer dtype, got {q.dtype}")
    if torch.any((q < -8) | (q > 7)):
        raise ValueError("q contains values outside signed int4 range [-8, 7]")
    rows, hidden = q.shape
    packed_width = (hidden + 1) // 2
    if out is None:
        out = torch.empty((rows, packed_width), dtype=torch.uint8, device=q.device)
    elif out.shape != (rows, packed_width) or out.dtype != torch.uint8:
        raise ValueError("out must be uint8 with shape [rows, ceil(hidden / 2)]")
    elif out.device != q.device:
        raise ValueError("q and out must be on the same device")

    pairs = torch.zeros((rows, packed_width * 2), dtype=torch.int16, device=q.device)
    pairs[:, :hidden] = q.to(torch.int16)
    lo = pairs[:, 0::2] & 0xF
    hi = pairs[:, 1::2] & 0xF
    out.copy_((lo | (hi << 4)).to(torch.uint8))
    return out


def unpack_int4(packed: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Unpack two's-complement int4 bytes into signed int8 values."""
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed must be a rank-2 uint8 tensor")
    if hidden_size < 0 or packed.shape[1] != (hidden_size + 1) // 2:
        raise ValueError("packed width does not match hidden_size")
    values = packed.to(torch.int16)
    lo = values & 0xF
    hi = (values >> 4) & 0xF
    q = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    q = torch.where(q >= 8, q - 16, q)
    return q[:, :hidden_size].to(torch.int8)


def quantize_and_pack_int4_reference(
    x: torch.Tensor,
    group_size: int = 128,
    out_q: Optional[torch.Tensor] = None,
    out_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch reference for symmetric groupwise int4 quantization."""
    rows, hidden, packed_width, num_groups = _validate_input(x, group_size)
    if (out_q is None) != (out_scales is None):
        raise ValueError("out_q and out_scales must be supplied together")
    if out_q is None:
        out_q = torch.empty((rows, packed_width), dtype=torch.uint8, device=x.device)
        out_scales = torch.empty(
            (rows, num_groups), dtype=torch.float16, device=x.device
        )
    else:
        _validate_outputs(x, out_q, out_scales, packed_width, num_groups)

    padded_hidden = num_groups * group_size
    padded = torch.zeros((rows, padded_hidden), dtype=torch.float32, device=x.device)
    padded[:, :hidden] = x.to(torch.float32)
    grouped = padded.reshape(rows, num_groups, group_size)
    amax = grouped.abs().amax(dim=-1)
    scales = torch.where(amax == 0, torch.ones_like(amax), amax / 7.0)
    stored_scales = scales.to(torch.float16)
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
    out_scales.copy_(stored_scales)
    pack_int4(q.reshape(rows, padded_hidden)[:, :hidden], out_q)
    return out_q, out_scales


def quantize_and_pack_int4(
    x: torch.Tensor,
    group_size: int = 128,
    out_q: Optional[torch.Tensor] = None,
    out_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` as symmetric groupwise int4 and pack two values per byte.

    The output is ``(q uint8 [rows, ceil(hidden / 2)], scales float16
    [rows, ceil(hidden / group_size)])``.  Supplying both output tensors writes
    into caller-owned storage, suitable for CUDA graph capture.
    """
    rows, _, packed_width, num_groups = _validate_input(x, group_size)
    if (out_q is None) != (out_scales is None):
        raise ValueError("out_q and out_scales must be supplied together")
    if out_q is None:
        out_q = torch.empty((rows, packed_width), dtype=torch.uint8, device=x.device)
        out_scales = torch.empty(
            (rows, num_groups), dtype=torch.float16, device=x.device
        )
    else:
        _validate_outputs(x, out_q, out_scales, packed_width, num_groups)

    if not x.is_cuda:
        return quantize_and_pack_int4_reference(x, group_size, out_q, out_scales)

    from sglang.kernels.ops.moe_trace.int4_pack import quantize_and_pack_int4_cuda

    return quantize_and_pack_int4_cuda(x, group_size, out_q, out_scales)


def dequantize_packed_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    hidden_size: int,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Decode packed signed int4 values using per-row, per-group scales."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if group_size % 2:
        raise ValueError("group_size must be even for packed int4 quantization")
    if scales.ndim != 2 or scales.dtype != torch.float16:
        raise ValueError("scales must be a rank-2 float16 tensor")
    if packed.shape[0] != scales.shape[0] or packed.device != scales.device:
        raise ValueError("packed and scales must have matching rows and devices")
    expected_groups = (hidden_size + group_size - 1) // group_size
    if scales.shape[1] != expected_groups:
        raise ValueError("scales width does not match hidden_size and group_size")
    q = unpack_int4(packed, hidden_size).to(torch.float32)
    group_indices = torch.arange(hidden_size, device=packed.device) // group_size
    return (q * scales.to(torch.float32)[:, group_indices]).to(dtype)


__all__ = [
    "dequantize_packed_int4",
    "pack_int4",
    "quantize_and_pack_int4",
    "quantize_and_pack_int4_reference",
    "unpack_int4",
]
