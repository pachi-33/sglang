"""Public FP16 dense operations for the stateless Qwen3.5 path."""

import torch

from .kernels.dense import embedding as _embedding
from .kernels.dense import nt_gemm
from .weights import Weight


def linear_fp16(
    x: torch.Tensor, weight: Weight, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply an FP16 ``[N,K]`` checkpoint matrix using Triton."""
    if weight.kind != "fp16" or len(weight.logical_shape) != 2:
        raise ValueError("linear_fp16 requires a rank-2 fp16 Weight")
    if x.ndim != 2 or x.shape[1] != weight.k:
        raise ValueError(f"expected x [M,{weight.k}], got {tuple(x.shape)}")
    return nt_gemm(x, weight.data, out)


def fp16_embedding(
    ids: torch.Tensor,
    table: Weight | torch.Tensor,
    out: torch.Tensor | None = None,
    validate_ids: bool = False,
) -> torch.Tensor:
    """Triton embedding lookup; invalid ids yield zero unless debug validation is requested."""
    data = table.data if isinstance(table, Weight) else table
    if isinstance(table, Weight) and (
        table.kind != "fp16" or len(table.logical_shape) != 2
    ):
        raise ValueError("embedding requires rank-2 fp16 Weight")
    return _embedding(ids, data, out, validate_ids)
