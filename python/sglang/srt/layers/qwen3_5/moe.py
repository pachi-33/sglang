"""V100-safe raw-NVFP4 routed MoE execution for Qwen3.5."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import triton

from .dense import linear_fp16
from .kernels.dense import nt_gemm
from .kernels.moe import (
    dispatch_block_expert_kernel,
    dispatch_count_kernel,
    dispatch_init_kernel,
    dispatch_inverse_kernel,
    dispatch_prefix_kernel,
    dispatch_stable_scatter_kernel,
    expert_row_global_kernel,
    fp16_add_kernel,
    fp16_grouped_gemm_kernel,
    fp16_sigmoid_multiply_kernel,
    fp16_swiglu_kernel,
    gather_nvfp4_activation_kernel,
    normalize_topk_weights_kernel,
    normalized_top8_kernel,
    nvfp4_grouped_gemm_kernel,
    nvfp4_paired_gemm1_swiglu_a4_kernel,
    route_combine_fused_kernel,
    route_combine_kernel,
    route_combine_weighted_kernel,
    route_weight_fp16_kernel,
)
from .quantization import quantize_nvfp4
from .weights import Weight


@dataclass
class MoeWeights:
    """The Qwen3.5 MoE public weight contract.

    Fields use the common ``Weight`` ABI supplied by ``qwen3_5.weights``.  Raw
    routed gate/up is `[E,1024,2048]` logical (gate then up) and down is
    `[E,2048,512]`; FP16 router/shared fields are optional for layer-only tests.
    """

    router: Any | None = None
    gate_up: Any | None = None
    down: Any | None = None
    shared_gate_up: Any | None = None
    shared_down: Any | None = None
    shared_gate: Any | None = None
    # Offloaded layers keep routed tensors out of their layer dictionaries.
    # The store is intentionally duck-typed here so the resident MoE module
    # does not acquire an import-time dependency on the file runtime.
    expert_store: Any | None = None
    layer_id: int | None = None


@dataclass(frozen=True)
class ExpertActivation:
    """Internal expert-major A4 payload with its physical routing metadata."""

    data: torch.Tensor
    logical_shape: tuple[int, int]
    block_scale: torch.Tensor
    expert_blocks: torch.Tensor
    global_scales: torch.Tensor


def _field(weight: Any, name: str) -> torch.Tensor:
    value = getattr(weight, name, None)
    if value is None:
        raise ValueError(f"NVFP4 weight is missing {name}")
    return value


def _validate_nvfp4_expert_call(
    x: torch.Tensor,
    gate_up: Any,
    down: Any,
    ids: torch.Tensor,
    weights: torch.Tensor,
    expert_to_slot: torch.Tensor | None = None,
) -> None:
    """Validate the fixed CUDA ABI before any routing or kernel launch."""
    if (
        x.ndim != 2
        or x.dtype != torch.float16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous CUDA FP16 [T,H]")
    if (
        ids.dtype != torch.int32
        or weights.dtype != torch.float32
        or not ids.is_cuda
        or not weights.is_cuda
    ):
        raise ValueError("ids must be CUDA int32 and route weights CUDA float32")
    if (
        not ids.is_contiguous()
        or not weights.is_contiguous()
        or ids.shape != (x.shape[0], 8)
        or weights.shape != ids.shape
    ):
        raise ValueError("ids and route weights must be contiguous [T,8]")
    if ids.device != x.device or weights.device != x.device:
        raise ValueError("x, ids, and route weights must share a CUDA device")
    if x.shape[1] <= 0 or x.shape[1] % 32:
        raise ValueError(
            "NVFP4 routed hidden size must be positive and divisible by 32"
        )
    gate_data, gate_sf = _field(gate_up, "data"), _field(gate_up, "block_scale")
    down_data, down_sf = _field(down, "data"), _field(down, "block_scale")
    if gate_data.ndim != 3 or down_data.ndim != 3:
        raise ValueError("NVFP4 routed weights must be rank-3 expert tensors")
    physical_experts = gate_data.shape[0]
    if physical_experts <= 0 or down_data.shape[0] != physical_experts:
        raise ValueError("NVFP4 gate/up and down must have the same physical experts")
    if expert_to_slot is None and physical_experts != 256:
        raise ValueError("resident NVFP4 routed weights must have 256 physical experts")
    intermediate = gate_data.shape[1] // 2
    if intermediate <= 0 or intermediate % 32:
        raise ValueError(
            "NVFP4 routed intermediate size must be positive and divisible by 32"
        )
    if gate_data.shape != (
        physical_experts,
        intermediate * 2,
        x.shape[1] // 2,
    ) or down_data.shape != (physical_experts, x.shape[1], intermediate // 2):
        raise ValueError("NVFP4 routed shapes must be gate/up [C,2I,H], down [C,H,I]")
    expected_gate_sf = (physical_experts, intermediate * 2, x.shape[1] // 16)
    expected_down_sf = (physical_experts, x.shape[1], intermediate // 16)
    tensors = (
        (gate_data, torch.uint8, "gate/up data"),
        (gate_sf, torch.uint8, "gate/up scales"),
        (down_data, torch.uint8, "down data"),
        (down_sf, torch.uint8, "down scales"),
        (_field(gate_up, "global_scale"), torch.float32, "gate/up weight globals"),
        (_field(gate_up, "input_global_scale"), torch.float32, "gate/up input globals"),
        (_field(down, "global_scale"), torch.float32, "down weight globals"),
        (_field(down, "input_global_scale"), torch.float32, "down input globals"),
    )
    if gate_sf.shape != expected_gate_sf or down_sf.shape != expected_down_sf:
        raise ValueError(
            "NVFP4 scale payload shape is inconsistent with routed matrices"
        )
    for tensor, dtype, label in tensors:
        if (
            tensor.dtype != dtype
            or not tensor.is_cuda
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{label} must be contiguous {dtype} on x.device")
    for name in ("global_scale", "input_global_scale"):
        if (
            _field(gate_up, name).numel() != physical_experts
            or _field(down, name).numel() != physical_experts
        ):
            raise ValueError("NVFP4 globals must be physical [C] tensors")
    if expert_to_slot is not None:
        if (
            expert_to_slot.shape != (256,)
            or expert_to_slot.dtype != torch.int32
            or not expert_to_slot.is_cuda
            or expert_to_slot.device != x.device
            or not expert_to_slot.is_contiguous()
        ):
            raise ValueError(
                "expert_to_slot must be contiguous CUDA int32 [256] on x.device"
            )
        if ids.numel():
            logical_min, logical_max = torch.aminmax(ids)
            if logical_min.item() < 0 or logical_max.item() >= 256:
                raise ValueError("route IDs must be logical experts in [0, 255]")
            used_slots = expert_to_slot[ids.to(torch.int64)]
            slot_min, slot_max = torch.aminmax(used_slots)
            if slot_min.item() < 0:
                raise ValueError("expert_to_slot is missing a routed expert")
            if slot_max.item() >= physical_experts:
                raise ValueError("expert_to_slot references a slot outside the cache")


def _validate_fp16_expert_call(
    x: torch.Tensor, gate_up: Any, down: Any, ids: torch.Tensor, weights: torch.Tensor
) -> None:
    """Validate FP16 routed tensors before grouped-GEMM indexing."""
    if (
        x.ndim != 2
        or x.dtype != torch.float16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous CUDA FP16 [T,H]")
    if (
        ids.dtype != torch.int32
        or weights.dtype != torch.float32
        or not ids.is_cuda
        or not weights.is_cuda
    ):
        raise ValueError("ids must be CUDA int32 and route weights CUDA float32")
    if (
        ids.shape != (x.shape[0], 8)
        or weights.shape != ids.shape
        or not ids.is_contiguous()
        or not weights.is_contiguous()
    ):
        raise ValueError("ids and route weights must be contiguous [T,8]")
    if ids.device != x.device or weights.device != x.device:
        raise ValueError("x, ids, and route weights must share a CUDA device")
    gate_data, down_data = _field(gate_up, "data"), _field(down, "data")
    if (
        gate_data.ndim != 3
        or down_data.ndim != 3
        or gate_data.shape[0] != 256
        or down_data.shape[0] != 256
    ):
        raise ValueError("FP16 routed weights must be [256,N,K]")
    intermediate = gate_data.shape[1] // 2
    if (
        intermediate <= 0
        or gate_data.shape != (256, intermediate * 2, x.shape[1])
        or down_data.shape != (256, x.shape[1], intermediate)
    ):
        raise ValueError(
            "FP16 routed shapes must be gate/up [256,2I,H], down [256,H,I]"
        )
    for tensor, label in ((gate_data, "gate/up data"), (down_data, "down data")):
        if (
            tensor.dtype != torch.float16
            or not tensor.is_cuda
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{label} must be contiguous CUDA FP16 on x.device")


def route_topk(
    logits: torch.Tensor, top_k: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k router with FP32 selected-weight renormalization and stable ties."""
    if logits.ndim != 2 or not 0 < top_k <= logits.shape[1]:
        raise ValueError("router logits must be [T,E] and top_k <= E")
    if (
        not logits.is_cuda
        or not logits.is_contiguous()
        or logits.shape[1] != 256
        or top_k != 8
    ):
        raise ValueError("V100 routed Top-8 requires contiguous CUDA logits [T,256]")
    tokens = logits.shape[0]
    ids = torch.empty((tokens, 8), dtype=torch.int32, device=logits.device)
    weights = torch.empty((tokens, 8), dtype=torch.float32, device=logits.device)
    if tokens:
        normalized_top8_kernel[(tokens,)](
            logits, ids, weights, tokens=tokens, experts=256, TOPK=8, num_warps=4
        )
        normalize_topk_weights_kernel[(tokens,)](
            weights, tokens=tokens, TOPK=8, num_warps=1
        )
    return ids, weights


def _build_dispatch(
    ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-only stable expert-major dispatch with fixed padded capacity."""
    if not ids.is_cuda or not ids.is_contiguous() or ids.dtype != torch.int32:
        raise ValueError("dispatch requires contiguous CUDA int32 route IDs")
    if num_experts != 256 or ids.shape[1] != 8:
        raise ValueError("V100 dispatch has a fixed 256-expert Top-8 contract")
    tokens, top_k = ids.shape
    device = ids.device
    routes = tokens * top_k
    max_blocks = triton.cdiv(routes, 32) + 256
    capacity = max_blocks * 32
    source = torch.empty((capacity,), device=device, dtype=torch.int32)
    positions = torch.empty((capacity,), device=device, dtype=torch.int32)
    expert_blocks = torch.empty((max_blocks,), device=device, dtype=torch.int32)
    counts = torch.empty((256,), device=device, dtype=torch.int32)
    offsets = torch.empty((257,), device=device, dtype=torch.int32)
    dispatch_init_kernel[(triton.cdiv(capacity, 256),)](
        source, positions, capacity=capacity, BLOCK=256
    )
    route_block = triton.next_power_of_2(max(routes, 1))
    dispatch_count_kernel[(256,)](
        ids, counts, routes=routes, ROUTE_BLOCK=route_block, BLOCK=256, num_warps=4
    )
    dispatch_prefix_kernel[(1,)](counts, offsets, num_warps=1)
    dispatch_block_expert_kernel[(max_blocks,)](
        offsets, expert_blocks, max_blocks=max_blocks, num_warps=1
    )
    if routes:
        dispatch_stable_scatter_kernel[(256,)](
            ids,
            source,
            positions,
            offsets,
            routes=routes,
            ROUTE_BLOCK=route_block,
            BLOCK=256,
            num_warps=4,
        )
    return source, positions, expert_blocks


def _grouped_gemm(
    activation,
    source_ids: torch.Tensor,
    expert_blocks: torch.Tensor,
    weight: Any,
    activation_global: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    route_weights: torch.Tensor | None = None,
    expert_major: bool = False,
    expert_to_slot: torch.Tensor | None = None,
) -> torch.Tensor:
    data = _field(weight, "data")
    scale = _field(weight, "block_scale")
    weight_global = _field(weight, "global_scale").reshape(-1).float()
    if data.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError(
            "NVFP4 grouped GEMM needs raw uint8 packed data and E4M3 scale bytes"
        )
    rows = activation.data.shape[0] if expert_major else source_ids.numel()
    num_experts, n, half_k = data.shape
    k = half_k * 2
    if rows == 0:
        return torch.empty((0, n), dtype=torch.float16, device=data.device)
    if rows % 32 or n % 32 or k % 32:
        raise ValueError("V100 NVFP4 kernel requires rows%32=N%32=K%32=0")
    ag = activation_global.reshape(-1).float()
    if ag.numel() != num_experts:
        raise ValueError(
            "V100 grouped GEMM requires physical per-expert activation globals"
        )
    if weight_global.numel() != num_experts:
        raise ValueError(
            "V100 grouped GEMM requires physical per-expert weight globals"
        )
    weighted = positions is not None or route_weights is not None
    if weighted != (positions is not None and route_weights is not None):
        raise ValueError("weighted grouped GEMM requires positions and route weights")
    if expert_major:
        if tuple(activation.data.shape) != (rows, k // 2) or tuple(
            activation.block_scale.shape
        ) != (rows, k // 16):
            raise ValueError("expert-major NVFP4 payload has invalid shape")
        gathered_data, gathered_sf = activation.data, activation.block_scale
    else:
        half_k, sf_k = k // 2, k // 16
        gathered_data = torch.empty(
            (rows, half_k), dtype=torch.uint8, device=data.device
        )
        gathered_sf = torch.empty((rows, sf_k), dtype=torch.uint8, device=data.device)
        gather_nvfp4_activation_kernel[(rows,)](
            activation.data,
            activation.block_scale,
            source_ids,
            gathered_data,
            gathered_sf,
            rows=rows,
            half_k=half_k,
            sf_k=sf_k,
            stride_src_m=activation.data.stride(0),
            stride_src_sf_m=activation.block_scale.stride(0),
            BLOCK=triton.next_power_of_2(half_k),
            num_warps=4,
        )
    out = torch.empty(
        (rows, n),
        dtype=torch.float32 if weighted else torch.float16,
        device=data.device,
    )
    grid = (rows // 32, triton.cdiv(n, 32))
    nvfp4_grouped_gemm_kernel[grid](
        gathered_data,
        gathered_sf,
        expert_blocks,
        data,
        scale,
        ag,
        weight_global,
        out,
        positions if weighted else expert_blocks,
        route_weights.reshape(-1) if weighted else expert_blocks,
        expert_to_slot if expert_to_slot is not None else expert_blocks,
        rows=rows,
        n=n,
        k=k,
        stride_am=gathered_data.stride(0),
        stride_as_m=gathered_sf.stride(0),
        stride_we=data.stride(0),
        stride_wn=data.stride(1),
        stride_ws_e=scale.stride(0),
        stride_ws_n=scale.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BM=32,
        BN=32,
        BK=32,
        WEIGHTED=weighted,
        USE_SLOT_MAP=expert_to_slot is not None,
        num_warps=4,
        num_stages=1,
    )
    return out


def _paired_gemm1_swiglu_a4(
    activation,
    source_ids: torch.Tensor,
    expert_blocks: torch.Tensor,
    gate_up: Any,
    down: Any,
    *,
    capture_z: bool = False,
    expert_to_slot: torch.Tensor | None = None,
) -> Any:
    """Fused expert GEMM1 pair, SwiGLU, and per-expert A4 quantization.

    The returned payload is raw NVFP4 directly consumable by GEMM2.  Unlike
    the old path, it never creates a full FP16 gate/up or SwiGLU tensor.
    """
    data = _field(gate_up, "data")
    scale = _field(gate_up, "block_scale")
    rows = source_ids.numel()
    experts, outputs, half_k = data.shape
    k = half_k * 2
    intermediate = outputs // 2
    if outputs != intermediate * 2 or intermediate % 32 or k % 32 or rows % 32:
        raise ValueError(
            "paired V100 GEMM1 requires [E,2I,K] with I/K/rows divisible by 32"
        )
    gate_a_global = _field(gate_up, "input_global_scale").reshape(-1).float()
    gate_w_global = _field(gate_up, "global_scale").reshape(-1).float()
    down_a_global = _field(down, "input_global_scale").reshape(-1).float()
    if any(g.numel() != experts for g in (gate_a_global, gate_w_global, down_a_global)):
        raise ValueError("fused expert path requires physical per-expert global scales")
    if rows == 0:
        payload = ExpertActivation(
            torch.empty((0, intermediate // 2), dtype=torch.uint8, device=data.device),
            (0, intermediate),
            torch.empty((0, intermediate // 16), dtype=torch.uint8, device=data.device),
            expert_blocks,
            down_a_global,
        )
        return (
            (
                payload,
                torch.empty((0, intermediate), dtype=torch.float16, device=data.device),
            )
            if capture_z
            else payload
        )

    z_data = torch.empty(
        (rows, intermediate // 2), dtype=torch.uint8, device=data.device
    )
    z_sf = torch.empty(
        (rows, intermediate // 16), dtype=torch.uint8, device=data.device
    )
    tiles = (rows // 32) * (intermediate // 32)
    # A bounded persistent grid: each CTA owns exactly one 32x32 (2 KiB)
    # scratch tile and walks its assigned output tiles serially.
    # The verified default is 320 V100 CTAs; benchmark tooling may select a
    # candidate cap without changing the mathematical kernel contract.
    cta_cap = int(os.environ.get("SGLANG_QWEN35_MOE_CTA_CAP", "320"))
    if cta_cap not in (80, 160, 320):
        raise ValueError("SGLANG_QWEN35_MOE_CTA_CAP must be one of 80, 160, or 320")
    programs = min(tiles, cta_cap)
    scratch = torch.empty((programs, 32, 32), dtype=torch.float16, device=data.device)
    z_capture = (
        torch.empty((rows, intermediate), dtype=torch.float16, device=data.device)
        if capture_z
        else scratch
    )
    max_tiles = triton.cdiv(tiles, programs)
    nvfp4_paired_gemm1_swiglu_a4_kernel[(programs,)](
        activation.data,
        activation.block_scale,
        source_ids,
        expert_blocks,
        data,
        scale,
        gate_a_global,
        gate_w_global,
        down_a_global,
        scratch,
        z_capture,
        z_data,
        z_sf,
        expert_to_slot if expert_to_slot is not None else expert_blocks,
        rows=rows,
        intermediate=intermediate,
        k=k,
        stride_am=activation.data.stride(0),
        stride_as_m=activation.block_scale.stride(0),
        stride_we=data.stride(0),
        stride_wn=data.stride(1),
        stride_ws_e=scale.stride(0),
        stride_ws_n=scale.stride(1),
        NUM_PROGRAMS=programs,
        MAX_TILES_PER_PROGRAM=max_tiles,
        CAPTURE_Z=capture_z,
        USE_SLOT_MAP=expert_to_slot is not None,
        num_warps=4,
        num_stages=1,
    )
    payload = ExpertActivation(
        z_data, (rows, intermediate), z_sf, expert_blocks, down_a_global
    )
    return (payload, z_capture) if capture_z else payload


def execute_experts(
    x: torch.Tensor,
    gate_up: Any,
    down: Any,
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    expert_to_slot: torch.Tensor | None = None,
    shared: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    capture_routes: bool = False,
) -> torch.Tensor:
    """Execute real W4A4 routed experts and deterministically combine top-k."""
    _validate_nvfp4_expert_call(
        x, gate_up, down, ids, weights, expert_to_slot=expert_to_slot
    )
    if shared is not None and (
        shared.shape != x.shape
        or shared.dtype != torch.float16
        or not shared.is_contiguous()
        or shared.device != x.device
    ):
        raise ValueError(
            "shared expert output must be contiguous FP16 [T,H] on x.device"
        )
    if residual is not None and (
        residual.shape != x.shape
        or residual.dtype != torch.float16
        or not residual.is_contiguous()
        or residual.device != x.device
    ):
        raise ValueError("residual must be contiguous FP16 [T,H] on x.device")
    gate_ga = _field(gate_up, "input_global_scale").reshape(-1).float()
    if gate_ga.numel() != _field(gate_up, "data").shape[0]:
        raise ValueError("gate/up input globals must be physically present per expert")
    if x.shape[0] == 0:
        out = torch.zeros_like(x)
        return (
            (
                out,
                torch.empty((0, x.shape[1]), dtype=torch.float32, device=x.device),
                torch.empty((0,), dtype=torch.int32, device=x.device),
            )
            if capture_routes
            else out
        )
    source, positions, expert_blocks = _build_dispatch(ids, 256)
    if source.numel() == 0:
        return torch.zeros_like(x)
    if expert_to_slot is None:
        first_gate_scale = gate_ga[0]
    else:
        first_logical_expert = ids.reshape(-1)[0].to(torch.int64)
        first_physical_expert = expert_to_slot[first_logical_expert].to(torch.int64)
        first_gate_scale = gate_ga[first_physical_expert]
    qx = quantize_nvfp4(x, first_gate_scale)
    down_ga = _field(down, "input_global_scale").reshape(-1).float()
    qz = _paired_gemm1_swiglu_a4(
        qx,
        source,
        expert_blocks,
        gate_up,
        down,
        expert_to_slot=expert_to_slot,
    )
    down_out = _grouped_gemm(
        qz,
        None,
        expert_blocks,
        down,
        down_ga,
        positions=positions,
        route_weights=weights,
        expert_major=True,
        expert_to_slot=expert_to_slot,
    )
    inverse = torch.empty((ids.numel(),), device=x.device, dtype=torch.int32)
    dispatch_inverse_kernel[(triton.cdiv(positions.numel(), 256),)](
        positions, inverse, rows=positions.numel(), routes=ids.numel(), BLOCK=256
    )
    out = torch.empty_like(x)
    route_combine_fused_kernel[(x.shape[0], triton.cdiv(x.shape[1], 256))](
        down_out,
        inverse,
        out,
        shared if shared is not None else out,
        residual if residual is not None else out,
        tokens=x.shape[0],
        hidden=x.shape[1],
        stride_down_m=down_out.stride(0),
        BLOCK=256,
        HAS_SHARED=shared is not None,
        HAS_RESIDUAL=residual is not None,
        num_warps=4,
    )
    return (out, down_out, inverse) if capture_routes else out


def execute_experts_unfused_baseline(
    x: torch.Tensor, gate_up: Any, down: Any, ids: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Same-semantic Triton baseline retaining materialized GEMM1/SwiGLU A4."""
    _validate_nvfp4_expert_call(x, gate_up, down, ids, weights)
    if x.shape[0] == 0:
        return torch.zeros_like(x)
    source, positions, expert_blocks = _build_dispatch(ids, 256)
    gate_ga = _field(gate_up, "input_global_scale").reshape(-1)
    qx = quantize_nvfp4(x, gate_ga[0])
    gate_up_out = _grouped_gemm(qx, source, expert_blocks, gate_up, gate_ga)
    z = _fp16_swiglu_values(gate_up_out)
    down_ga = _field(down, "input_global_scale").reshape(-1)
    row_ga = torch.empty((source.numel(),), dtype=torch.float32, device=x.device)
    expert_row_global_kernel[(triton.cdiv(source.numel(), 256),)](
        expert_blocks, down_ga, row_ga, rows=source.numel(), BLOCK=256, num_warps=4
    )
    qz = quantize_nvfp4(z, row_ga)
    expert_z = ExpertActivation(
        qz.data, qz.logical_shape, qz.block_scale, expert_blocks, down_ga
    )
    down_out = _grouped_gemm(
        expert_z,
        None,
        expert_blocks,
        down,
        down_ga,
        positions=positions,
        route_weights=weights,
        expert_major=True,
    )
    inverse = torch.empty((ids.numel(),), device=x.device, dtype=torch.int32)
    dispatch_inverse_kernel[(triton.cdiv(positions.numel(), 256),)](
        positions, inverse, rows=positions.numel(), routes=ids.numel(), BLOCK=256
    )
    out = torch.empty_like(x)
    route_combine_fused_kernel[(x.shape[0], triton.cdiv(x.shape[1], 256))](
        down_out,
        inverse,
        out,
        out,
        out,
        tokens=x.shape[0],
        hidden=x.shape[1],
        stride_down_m=down_out.stride(0),
        HAS_SHARED=False,
        HAS_RESIDUAL=False,
        BLOCK=256,
        num_warps=4,
    )
    return out


def _fp16_linear(x: torch.Tensor, weight: Any) -> torch.Tensor:
    if isinstance(weight, Weight):
        return linear_fp16(x, weight)
    # Lightweight containers used by layer tests retain the public matrix ABI.
    w = _field(weight, "data") if hasattr(weight, "data") else weight
    return nt_gemm(x, w.to(dtype=x.dtype))


def _fp16_swiglu_values(gate_up: torch.Tensor) -> torch.Tensor:
    rows, outputs = gate_up.shape
    intermediate = outputs // 2
    if outputs != intermediate * 2:
        raise ValueError("SwiGLU input must be [M,2I]")
    z = torch.empty((rows, intermediate), dtype=torch.float16, device=gate_up.device)
    if rows:
        fp16_swiglu_kernel[(rows, triton.cdiv(intermediate, 32))](
            gate_up,
            z,
            rows=rows,
            intermediate=intermediate,
            stride_gu_m=gate_up.stride(0),
            stride_z_m=z.stride(0),
            num_warps=1,
        )
    return z


def _fp16_sigmoid_multiply(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if (
        gate.ndim != 2
        or gate.shape[0] != value.shape[0]
        or gate.shape[1] not in (1, value.shape[1])
    ):
        raise ValueError("shared gate and value shapes must match")
    rows, hidden = value.shape
    out = torch.empty_like(value)
    if rows:
        fp16_sigmoid_multiply_kernel[(rows, triton.cdiv(hidden, 256))](
            value,
            gate,
            out,
            rows=rows,
            hidden=hidden,
            stride_value_m=value.stride(0),
            stride_gate_m=gate.stride(0),
            stride_out_m=out.stride(0),
            GATE_SCALAR=gate.shape[1] == 1,
            BLOCK=256,
            num_warps=4,
        )
    return out


def _fp16_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape or a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("FP16 residual add requires matching FP16 tensors")
    rows, hidden = a.shape
    out = torch.empty_like(a)
    if rows:
        fp16_add_kernel[(rows, triton.cdiv(hidden, 256))](
            a,
            b,
            out,
            rows=rows,
            hidden=hidden,
            stride_a_m=a.stride(0),
            stride_b_m=b.stride(0),
            stride_out_m=out.stride(0),
            BLOCK=256,
            num_warps=4,
        )
    return out


def _fp16_grouped_gemm(
    activation: torch.Tensor,
    source_ids: torch.Tensor | None,
    expert_blocks: torch.Tensor,
    weight: Any,
    *,
    identity_source: bool = False,
) -> torch.Tensor:
    data = _field(weight, "data")
    if data.ndim != 3 or data.dtype != torch.float16:
        raise ValueError("routed FP16 weights must be [E,N,K] FP16")
    rows = activation.shape[0] if identity_source else source_ids.numel()
    experts, n, k = data.shape
    if activation.ndim != 2 or activation.shape[1] != k or rows % 32:
        raise ValueError("FP16 grouped GEMM requires [M,K] and a 32-row dispatch")
    if expert_blocks.numel() != rows // 32 or experts != 256:
        raise ValueError("FP16 routed GEMM requires 256 physical experts")
    out = torch.empty((rows, n), dtype=torch.float16, device=activation.device)
    fp16_grouped_gemm_kernel[(rows // 32, triton.cdiv(n, 32))](
        activation,
        activation if identity_source else source_ids,
        expert_blocks,
        data,
        out,
        rows=rows,
        n=n,
        k=k,
        stride_am=activation.stride(0),
        stride_we=data.stride(0),
        stride_wn=data.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        IDENTITY_SOURCE=identity_source,
        num_warps=4,
        num_stages=1,
    )
    return out


def _fp16_swiglu(x: torch.Tensor, gate_up: Any, down: Any) -> torch.Tensor:
    """One FP16 gate/up/down expert (also used for the shared expert)."""
    gu = _fp16_linear(x, gate_up).to(torch.float16)
    z = _fp16_swiglu_values(gu)
    return _fp16_linear(z, down).to(x.dtype)


def execute_fp16_experts(
    x: torch.Tensor,
    gate_up: Any,
    down: Any,
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    capture_routes: bool = False,
) -> torch.Tensor:
    """Deterministic top-k FP16 expert path for Qwen3.5 layers 0 and 39."""
    _validate_fp16_expert_call(x, gate_up, down, ids, weights)
    if x.shape[0] == 0:
        out = torch.zeros_like(x)
        return (
            (
                out,
                torch.empty((0, x.shape[1]), dtype=torch.float32, device=x.device),
                torch.empty((0,), dtype=torch.int32, device=x.device),
            )
            if capture_routes
            else out
        )
    source, positions, expert_blocks = _build_dispatch(ids, 256)
    gate_up_out = _fp16_grouped_gemm(x, source, expert_blocks, gate_up)
    z = _fp16_swiglu_values(gate_up_out)
    down_out = _fp16_grouped_gemm(z, None, expert_blocks, down, identity_source=True)
    inverse = torch.empty((ids.numel(),), device=x.device, dtype=torch.int32)
    dispatch_inverse_kernel[(triton.cdiv(positions.numel(), 256),)](
        positions, inverse, rows=positions.numel(), routes=ids.numel(), BLOCK=256
    )
    out = torch.empty_like(x)
    route_combine_kernel[(x.shape[0], triton.cdiv(x.shape[1], 256))](
        down_out,
        inverse,
        weights,
        out,
        tokens=x.shape[0],
        hidden=x.shape[1],
        stride_down_m=down_out.stride(0),
        BLOCK=256,
        num_warps=4,
    )
    if not capture_routes:
        return out
    weighted_routes = torch.empty(
        (source.numel(), x.shape[1]), dtype=torch.float32, device=x.device
    )
    route_weight_fp16_kernel[(source.numel(), triton.cdiv(x.shape[1], 256))](
        down_out,
        positions,
        weights.reshape(-1),
        weighted_routes,
        rows=source.numel(),
        hidden=x.shape[1],
        stride_down_m=down_out.stride(0),
        BLOCK=256,
        num_warps=4,
    )
    return out, weighted_routes, inverse


def fused_fp16_moe(
    x: torch.Tensor, router: Any, gate_up: Any, down: Any, top_k: int = 8
) -> torch.Tensor:
    """FP16 routed MoE fallback for layers 0/39; it still uses all 256 experts."""
    ids, probs = route_topk(_fp16_linear(x, router), top_k)
    return execute_fp16_experts(x, gate_up, down, ids, probs)


def fused_nvfp4_moe(
    x: torch.Tensor,
    weights: MoeWeights,
    top_k: int = 8,
    residual: torch.Tensor | None = None,
    *,
    capture_router: bool = False,
    expert_to_slot: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Router + routed W4A4 MoE + optional FP16 shared expert.

    ``capture_router`` is an opt-in validation hook.  It returns the normal
    result along with the selected Top-8 expert IDs and their normalized
    probabilities, without changing either the route or expert execution.
    The route tensors stay on CUDA; callers that need host-side diagnostics
    explicitly choose when to copy them out.
    """
    if weights.router is None:
        raise ValueError("router weight is required")
    offloaded = weights.expert_store is not None
    if offloaded:
        if weights.gate_up is not None or weights.down is not None:
            raise ValueError(
                "offloaded MoE must not retain resident routed expert weights"
            )
        if (
            isinstance(weights.layer_id, bool)
            or not isinstance(weights.layer_id, int)
            or not 1 <= weights.layer_id <= 38
        ):
            raise ValueError("offloaded MoE requires an original layer_id in 1..38")
        if expert_to_slot is not None:
            raise ValueError("offloaded MoE obtains expert_to_slot from its lease")
    elif weights.gate_up is None or weights.down is None:
        raise ValueError("resident MoE requires gate_up and down weights")
    if residual is not None and (
        residual.shape != x.shape
        or residual.dtype != torch.float16
        or not residual.is_contiguous()
        or residual.device != x.device
    ):
        raise ValueError("residual must be contiguous FP16 [T,H] on x.device")
    ids, probs = route_topk(_fp16_linear(x, weights.router), top_k)
    shared = None
    if weights.shared_gate_up is not None and weights.shared_down is not None:
        shared = _fp16_swiglu(x, weights.shared_gate_up, weights.shared_down)
        if weights.shared_gate is not None:
            shared = _fp16_sigmoid_multiply(
                shared, _fp16_linear(x, weights.shared_gate)
            )
    if not offloaded and getattr(weights.gate_up, "kind", None) == "fp16":
        out = execute_fp16_experts(x, weights.gate_up, weights.down, ids, probs)
        if shared is not None:
            out = _fp16_add(out, shared)
        if residual is not None:
            out = _fp16_add(out, residual)
    elif offloaded:
        # acquire performs the sole route D2H copy, then waits the compute
        # stream on every H2D-ready event before returning the complete map.
        # Leaving the context records a last-use event after both GEMMs and
        # combine have been enqueued, so no selected cache slot can be reused
        # while this layer is still consuming it.
        with weights.expert_store.acquire(weights.layer_id, ids) as lease:
            out = execute_experts(
                x,
                lease.gate_up,
                lease.down,
                ids,
                probs,
                expert_to_slot=lease.expert_to_slot,
                shared=shared,
                residual=residual,
            )
    else:
        out = execute_experts(
            x,
            weights.gate_up,
            weights.down,
            ids,
            probs,
            expert_to_slot=expert_to_slot,
            shared=shared,
            residual=residual,
        )
    if capture_router:
        return out, ids, probs
    return out


# Public name used by the model wrapper/reference harness.
fused_moe = fused_nvfp4_moe
