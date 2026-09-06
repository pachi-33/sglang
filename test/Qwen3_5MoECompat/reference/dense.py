"""Reference dense/norm operations with explicit Qwen3.5 rounding points."""

import torch


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps) * (1 + weight.float())).to(x.dtype)


def gated_rms_norm_silu(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps) * weight.float() * torch.nn.functional.silu(gate.float())).to(x.dtype)


def fp16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.float() @ weight.float().t()).to(torch.float16)
