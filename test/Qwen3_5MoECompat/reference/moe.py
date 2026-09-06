"""Streaming, independent routed-MoE reference used by the layer scan.

The routines here deliberately use the test codec instead of the runtime
quantizer.  They only expand one routed expert matrix at a time.
"""

from __future__ import annotations

import torch

from .codec import decode_e2m1, decode_e4m3fn, unpack_a4


def _a4_roundtrip(
    x: torch.Tensor, global_scale: torch.Tensor, quantize_a4
) -> torch.Tensor:
    """Run the supplied CPU codec and return the decoded FP32 CUDA operand."""
    data, scale, _ = quantize_a4(x.detach().cpu(), global_scale.detach().cpu())
    return unpack_a4(data, scale, global_scale.detach().cpu()).to(
        x.device, dtype=torch.float32
    )


def _decode_local_a4(
    data: torch.Tensor, scale: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Decode E2M1×E4M3 local operands without applying a tensor global.

    The NVFP4 Triton GEMMs multiply the reciprocal globals after the FP32
    accumulation.  Moving either global into an operand changes FP32 rounding
    before the required FP16 boundary and can flip a later A4 codeword.
    """
    packed = data.detach().cpu()
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    local = decode_e2m1(codes).float() * decode_e4m3fn(
        scale.detach().cpu()
    ).float().repeat_interleave(16, dim=1)
    return local.to(device=device, dtype=torch.float16)


def _decoded_expert(weight, expert: int) -> torch.Tensor:
    return unpack_a4(
        weight.data[expert].detach().cpu(),
        weight.block_scale[expert].detach().cpu(),
        weight.global_scale[expert : expert + 1].detach().cpu(),
    ).to(weight.data.device, dtype=torch.float32)


def stable_top8(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable Top-8 and selected-softmax renormalization in the checkpoint ABI."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    ids = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[..., :8]
    selected = probabilities.gather(-1, ids)
    return ids.to(torch.int32), selected / selected.sum(dim=-1, keepdim=True)


def routed_experts(
    x: torch.Tensor,
    gate_up,
    down,
    ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    quantize_a4,
    return_routes: bool = False,
) -> torch.Tensor:
    """Reference routed output with FP16 projection/SwiGLU/down boundaries."""
    routes_by_slot = torch.zeros(
        (x.shape[0], 8, x.shape[1]), dtype=torch.float32, device=x.device
    )
    # Retain raw local operands through each GEMM and apply global factors to
    # its FP32 accumulator, matching the frozen Volta kernel contract.
    x_data, x_scale, _ = quantize_a4(
        x.detach().cpu(), gate_up.input_global_scale[:1].detach().cpu()
    )
    x_local = _decode_local_a4(x_data, x_scale, device=x.device)
    for expert in range(256):
        routes = (ids == expert).nonzero(as_tuple=False)
        if routes.numel() == 0:
            continue
        for token, slot in routes.tolist():
            w1 = _decode_local_a4(
                gate_up.data[expert], gate_up.block_scale[expert], device=x.device
            )
            inv1 = 1.0 / (
                gate_up.input_global_scale[expert].float()
                * gate_up.global_scale[expert].float()
            )
            gu = ((x_local[token : token + 1].float() @ w1.float().t()) * inv1).to(
                torch.float16
            )
            del w1
            middle = (
                torch.nn.functional.silu(gu[:, : gu.shape[1] // 2].float())
                * gu[:, gu.shape[1] // 2 :].float()
            ).to(torch.float16)
            z_data, z_scale, _ = quantize_a4(
                middle.detach().cpu(),
                down.input_global_scale[expert : expert + 1].detach().cpu(),
            )
            z_local = _decode_local_a4(z_data, z_scale, device=x.device)
            w2 = _decode_local_a4(
                down.data[expert], down.block_scale[expert], device=x.device
            )
            inv2 = 1.0 / (
                down.input_global_scale[expert].float()
                * down.global_scale[expert].float()
            )
            value = ((z_local.float() @ w2.float().t()) * inv2).to(torch.float16)
            del w2
            routes_by_slot[token, slot] = (
                value[0].float() * route_weights[token, slot].float()
            )
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(8):
        result += routes_by_slot[:, slot]
    result = result.to(torch.float16)
    return (result, routes_by_slot) if return_routes else result


def fp16_routed_experts(
    x: torch.Tensor,
    gate_up,
    down,
    ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    return_routes: bool = False,
) -> torch.Tensor:
    routes_by_slot = torch.zeros(
        (x.shape[0], 8, x.shape[1]), dtype=torch.float32, device=x.device
    )
    for expert in range(256):
        routes = (ids == expert).nonzero(as_tuple=False)
        if routes.numel() == 0:
            continue
        for token, slot in routes.tolist():
            gu = (x[token : token + 1].float() @ gate_up.data[expert].float().t()).to(
                torch.float16
            )
            middle = (
                torch.nn.functional.silu(gu[:, :512].float()) * gu[:, 512:].float()
            ).to(torch.float16)
            value = (middle.float() @ down.data[expert].float().t()).to(torch.float16)
            routes_by_slot[token, slot] = (
                value[0].float() * route_weights[token, slot].float()
            )
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(8):
        result += routes_by_slot[:, slot]
    result = result.to(torch.float16)
    return (result, routes_by_slot) if return_routes else result


def shared_expert(x: torch.Tensor, gate_up, down, gate) -> torch.Tensor:
    gu = (x.float() @ gate_up.data.float().t()).to(torch.float16)
    z = (torch.nn.functional.silu(gu[:, :512].float()) * gu[:, 512:].float()).to(
        torch.float16
    )
    value = (z.float() @ down.data.float().t()).to(torch.float16)
    scalar = (x.float() @ gate.data.float().t()).to(torch.float16)
    return (value.float() * torch.sigmoid(scalar.float()).to(torch.float16).float()).to(
        torch.float16
    )


def moe(
    x: torch.Tensor,
    router,
    gate_up,
    down,
    shared_gate_up,
    shared_down,
    shared_gate,
    *,
    quantize_a4,
):
    logits = (x.float() @ router.data.float().t()).to(torch.float16)
    ids, weights = stable_top8(logits)
    routed = (
        fp16_routed_experts(x, gate_up, down, ids, weights)
        if gate_up.kind == "fp16"
        else routed_experts(x, gate_up, down, ids, weights, quantize_a4=quantize_a4)
    )
    shared = shared_expert(x, shared_gate_up, shared_down, shared_gate)
    return (routed.float() + shared.float()).to(torch.float16), logits, ids, weights
