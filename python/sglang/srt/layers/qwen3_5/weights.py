"""Typed compressed tensors used by the Qwen3.5 compatibility kernels."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


WeightKind = Literal["fp16", "fp8", "nvfp4"]
ActivationKind = Literal["fp8", "nvfp4"]


@dataclass(frozen=True)
class Weight:
    """A matrix or expert batch without an expanded copy of its weights.

    ``logical_shape`` always ends in ``(N, K)``.  FP8 ``data`` is the raw
    E4M3FN byte representation, and NVFP4 ``data`` packs even K in the low
    nibble.  The latter convention is part of the public checkpoint contract.
    """

    kind: WeightKind
    data: torch.Tensor
    logical_shape: Tuple[int, ...]
    block_scale: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    input_global_scale: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.kind not in ("fp16", "fp8", "nvfp4"):
            raise ValueError(f"unsupported weight kind: {self.kind}")
        if len(self.logical_shape) not in (2, 3):
            raise ValueError("logical_shape must end in (N, K)")
        n, k = self.logical_shape[-2:]
        if n <= 0 or k <= 0:
            raise ValueError(f"invalid logical matrix shape: {self.logical_shape}")
        if any(dim <= 0 for dim in self.logical_shape):
            raise ValueError(f"logical shape dimensions must be positive: {self.logical_shape}")
        prefix = self.logical_shape[:-2]
        expected = prefix + ((n, k) if self.kind != "nvfp4" else (n, (k + 1) // 2))
        if tuple(self.data.shape) != expected:
            raise ValueError(
                f"{self.kind} storage shape {tuple(self.data.shape)} != {expected}"
            )
        if self.kind == "fp16":
            if self.data.dtype != torch.float16:
                raise TypeError("fp16 weights must have torch.float16 dtype")
            if any(x is not None for x in (self.block_scale, self.global_scale, self.input_global_scale)):
                raise ValueError("fp16 weights cannot have quantization scales")
        elif self.kind == "fp8":
            if self.data.dtype != torch.uint8 or self.block_scale is None:
                raise TypeError("fp8 weights require uint8 data and block_scale")
            expected_scale = prefix + ((n + 127) // 128, (k + 127) // 128)
            if tuple(self.block_scale.shape) != expected_scale:
                raise ValueError(f"fp8 scale shape {tuple(self.block_scale.shape)} != {expected_scale}")
            if self.block_scale.dtype not in (torch.float16, torch.float32):
                raise TypeError("fp8 block_scale must be float16 or float32")
            if self.global_scale is not None or self.input_global_scale is not None:
                raise ValueError("fp8 weights use only block_scale")
        else:
            if k % 2:
                raise ValueError("NVFP4 K must be even")
            if self.data.dtype != torch.uint8 or self.block_scale is None:
                raise TypeError("nvfp4 weights require uint8 packed data and block_scale")
            expected_scale = prefix + (n, (k + 15) // 16)
            if tuple(self.block_scale.shape) != expected_scale:
                raise ValueError(f"nvfp4 scale shape {tuple(self.block_scale.shape)} != {expected_scale}")
            if self.block_scale.dtype != torch.uint8:
                raise TypeError("nvfp4 block_scale must be raw E4M3FN uint8")
            if self.global_scale is None or self.input_global_scale is None:
                raise ValueError("nvfp4 weights require weight and input global scales")
            expert_count = 1 if len(self.logical_shape) == 2 else self.logical_shape[0]
            for scale, name in ((self.global_scale, "global_scale"), (self.input_global_scale, "input_global_scale")):
                if scale.dtype != torch.float32 or scale.numel() not in (1, expert_count):
                    raise ValueError(f"nvfp4 {name} must be float32 scalar/[1] or [E]")
        for tensor in (self.block_scale, self.global_scale, self.input_global_scale):
            if tensor is not None and tensor.device != self.data.device:
                raise ValueError("weight storage and scales must be on the same device")

    @property
    def n(self) -> int:
        return self.logical_shape[-2]

    @property
    def k(self) -> int:
        return self.logical_shape[-1]

    @property
    def experts(self) -> int:
        return 1 if len(self.logical_shape) == 2 else self.logical_shape[0]


@dataclass(frozen=True)
class QuantActivation:
    """Compressed activation supplied to a fused quantized GEMM."""

    kind: ActivationKind
    data: torch.Tensor
    logical_shape: Tuple[int, int]
    block_scale: torch.Tensor
    global_scale: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        m, k = self.logical_shape
        if self.kind == "fp8":
            if tuple(self.data.shape) != (m, k) or self.data.dtype != torch.uint8:
                raise ValueError("fp8 activation must be uint8 [M, K]")
            if tuple(self.block_scale.shape) != (m, (k + 127) // 128):
                raise ValueError("invalid fp8 activation scale shape")
            if self.block_scale.dtype != torch.float32:
                raise TypeError("fp8 activation scale must be float32")
            if self.global_scale is not None:
                raise ValueError("fp8 activation has no global scale")
        elif self.kind == "nvfp4":
            if k % 2 or tuple(self.data.shape) != (m, k // 2) or self.data.dtype != torch.uint8:
                raise ValueError("nvfp4 activation must be packed uint8 [M, K/2]")
            if tuple(self.block_scale.shape) != (m, (k + 15) // 16):
                raise ValueError("invalid nvfp4 activation scale shape")
            if self.global_scale is None:
                raise ValueError("nvfp4 activation requires input global scale")
            if self.block_scale.dtype != torch.uint8 or self.global_scale.dtype != torch.float32:
                raise TypeError("nvfp4 activation requires uint8 scale and float32 global scale")
            if self.global_scale.numel() not in (1, m):
                raise ValueError("nvfp4 activation global scale must be scalar/[1] or [M]")
        else:
            raise ValueError(f"unsupported activation kind: {self.kind}")
        if self.data.device != self.block_scale.device or (self.global_scale is not None and self.data.device != self.global_scale.device):
            raise ValueError("activation storage and scales must be on the same device")
