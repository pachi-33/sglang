"""Mathematical FP8 E4M3FN and NVFP4 E2M1 reference codecs.

This module intentionally does not call the production quantization package.
It is written for auditability and test inputs, not performance.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _e4m3_values() -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for sign in range(2):
        for exponent in range(16):
            for mantissa in range(8):
                # E4M3FN reserves only exp=15,mantissa=7 as NaN.
                if exponent == 15 and mantissa == 7:
                    continue
                magnitude = (
                    mantissa / 8.0 * 2.0**-6
                    if exponent == 0
                    else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
                )
                code = (sign << 7) | (exponent << 3) | mantissa
                values.append((code, -magnitude if sign else magnitude))
    return values


_E4M3 = _e4m3_values()
_E4M3_DECODE_ARRAY = np.full(256, np.nan, dtype=np.float32)
for _code, _value in _E4M3:
    _E4M3_DECODE_ARRAY[_code] = _value
_E4M3_POSITIVE = np.array(
    [value for code, value in _E4M3 if code < 128], dtype=np.float64
)
_E4M3_POSITIVE_CODES = np.array(
    [code for code, value in _E4M3 if code < 128], dtype=np.uint8
)


def decode_e4m3fn(data: torch.Tensor) -> torch.Tensor:
    codes = data.detach().to(device="cpu", dtype=torch.uint8).numpy()
    decoded = _E4M3_DECODE_ARRAY[codes]
    return torch.from_numpy(decoded.copy()).to(device=data.device).reshape(data.shape)


def _encode_positive_nearest_even(
    values: np.ndarray, candidates: np.ndarray, codes: np.ndarray
) -> np.ndarray:
    """Vectorized finite saturation and exact RNE selection for sorted values."""
    clipped = np.clip(values, candidates[0], candidates[-1])
    right = np.searchsorted(candidates, clipped, side="left")
    right = np.clip(right, 0, len(candidates) - 1)
    left = np.maximum(right - 1, 0)
    left_distance = clipped - candidates[left]
    right_distance = candidates[right] - clipped
    choose_right = right_distance < left_distance
    ties = right_distance == left_distance
    choose_right |= ties & ((codes[right] & 1) == 0) & ((codes[left] & 1) != 0)
    return np.where(choose_right, codes[right], codes[left])


def encode_e4m3fn(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1)
    sign = np.signbit(flat)
    magnitude = np.where(np.isnan(flat), 0.0, np.abs(flat))
    codes = _encode_positive_nearest_even(
        magnitude, _E4M3_POSITIVE, _E4M3_POSITIVE_CODES
    )
    codes = codes | (sign.astype(np.uint8) << 7)
    return torch.from_numpy(codes.copy()).to(device=values.device).reshape(values.shape)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1(data: torch.Tensor) -> torch.Tensor:
    code = data.to(torch.uint8)
    sign = torch.where((code & 8) != 0, -1.0, 1.0)
    mag = torch.empty_like(code, dtype=torch.float32)
    for index, value in enumerate(_E2M1):
        mag = torch.where((code & 7) == index, value, mag)
    return sign * mag


def encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1)
    sign = np.signbit(flat)
    magnitude = np.where(np.isnan(flat), 0.0, np.abs(flat))
    codes = _encode_positive_nearest_even(
        magnitude, np.asarray(_E2M1, dtype=np.float64), np.arange(8, dtype=np.uint8)
    )
    codes = codes | (sign.astype(np.uint8) << 3)
    return torch.from_numpy(codes.copy()).to(device=values.device).reshape(values.shape)


def quantize_a8(x: torch.Tensor, group_size: int = 128):
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError("A8 reference expects [M,K] with K divisible by group size")
    shaped = x.float().reshape(x.shape[0], -1, group_size)
    # Keep this as a float32 reciprocal multiply.  The production kernels use
    # the same frozen RN32 scale contract; division changes rare E4M3 ties.
    scale = shaped.abs().amax(dim=-1).float() * torch.tensor(
        1.0 / 448.0, dtype=torch.float32, device=shaped.device
    )
    # A8 retains an IEEE signed zero even when its whole K128 scale is zero.
    # This matches the CUDA producer/quantizer.  NVFP4 deliberately differs:
    # its zero local-scale payload is canonically all-zero nibbles.
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = shaped / safe[..., None]
    return encode_e4m3fn(normalized.reshape_as(x)), scale.float()


def quantize_a4(x: torch.Tensor, global_scale: torch.Tensor, group_size: int = 16):
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError("A4 reference expects [M,K] with K divisible by group size")
    g = global_scale.float().reshape(-1, 1)
    if g.shape[0] not in (1, x.shape[0]):
        raise ValueError("global_scale must be scalar or one value per row")
    u = x.float() * g
    shaped = u.reshape(x.shape[0], -1, group_size)
    scale = encode_e4m3fn(
        shaped.abs().amax(dim=-1).float()
        * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=shaped.device)
    )
    decoded_scale = decode_e4m3fn(scale)[..., None]
    normalized = torch.where(
        decoded_scale == 0, torch.zeros_like(shaped), shaped / decoded_scale
    )
    q = encode_e2m1(normalized.reshape_as(x))
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return packed, scale, g.squeeze(1)


def unpack_a4(
    data: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    low = data & 15
    high = data >> 4
    q = torch.stack((low, high), dim=-1).reshape(data.shape[0], -1)
    local = decode_e4m3fn(scale).repeat_interleave(16, dim=1)
    return decode_e2m1(q) * local / global_scale.float().reshape(-1, 1)
