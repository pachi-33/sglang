"""M3 independent real-layer scan.

Run all layers with::

  CUDA_VISIBLE_DEVICES=GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96 PYTHONPATH=python:. \
  /home/yaozhenyang/downloads/yes/envs/sglang-v100/bin/python -m \
  test.Qwen3_5MoECompat.integration.layer_scan --layers 0-39

Each layer is loaded through :class:`Qwen35Checkpoint` and released before
the next is read.  JSON records are intentionally useful outside unittest.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import sys
import unittest
from pathlib import Path
from test.Qwen3_5MoECompat.reference import model as reference_model
from test.Qwen3_5MoECompat.reference import moe as reference_moe
from test.Qwen3_5MoECompat.reference.gdn import (
    recurrent_vectorized as reference_recurrent,
)
from test.Qwen3_5MoECompat.unit.test_environment import V100_UUID, V100TestCase

import torch
import triton

from sglang.srt.layers.qwen3_5.attention import causal_gqa
from sglang.srt.layers.qwen3_5.checkpoint import Qwen35Checkpoint
from sglang.srt.layers.qwen3_5.dense import linear_fp16
from sglang.srt.layers.qwen3_5.gdn import (
    depthwise_conv4_silu,
    l2_normalize_qk,
    prepare_gates,
    recurrent_gdn,
)
from sglang.srt.layers.qwen3_5.model_ops import (
    full_qk_rope_gate,
    gated_attention_fp8,
    gated_gdn_fp8,
    split_gdn_qkv,
)
from sglang.srt.layers.qwen3_5.moe import (
    MoeWeights,
    _fp16_sigmoid_multiply,
    _fp16_swiglu,
    _fp16_swiglu_values,
    execute_experts,
    execute_fp16_experts,
    fused_moe,
    route_topk,
)
from sglang.srt.layers.qwen3_5.ops import gemma_rms_norm, residual_add_gemma_rms_norm
from sglang.srt.layers.qwen3_5.quantization import linear_fp8, quantize_fp8

DEFAULT_MODEL = Path(
    os.environ.get(
        "QWEN35_MODEL_DIR",
        "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
    )
)
DEFAULT_OUTPUT = Path(
    os.environ.get("QWEN35_LAYER_SCAN_OUTPUT", "/tmp/qwen35-layer-scan")
)
SEED = 20260906
BUDGETS = {"projection": 2e-3, "local": 5e-3, "gdn_recurrent": 1e-4}
MATH_CONTRACT = "qwen35-m3-fp8-block-semantic-v4"
_RUN_SOURCE_HASH: str | None = None


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = sorted(
        list((root / "python/sglang/srt/layers/qwen3_5").rglob("*.py"))
        + list((root / "test/Qwen3_5MoECompat/reference").glob("*.py"))
        + [Path(__file__)],
        key=lambda path: str(path.relative_to(root)),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    budget: float,
    *,
    coverage: int | None = None,
) -> dict:
    if tuple(actual.shape) != tuple(reference.shape):
        raise AssertionError(
            f"shape mismatch: actual {tuple(actual.shape)} != reference {tuple(reference.shape)}"
        )
    a, r = actual.float(), reference.float()
    error = (a - r).abs()
    denom = torch.linalg.vector_norm(r)
    nrmse = (
        (torch.linalg.vector_norm(a - r) / denom).item()
        if denom.item()
        else error.max().item()
    )
    flat = error.flatten()
    return {
        "shape": list(actual.shape),
        "nrmse": nrmse,
        "max": error.max().item() if flat.numel() else 0.0,
        "p99": torch.quantile(flat, 0.99).item() if flat.numel() else 0.0,
        "finite": bool(
            torch.isfinite(actual).all().item()
            and torch.isfinite(reference).all().item()
        ),
        "budget": budget,
        "pass": bool(
            nrmse <= budget
            and torch.isfinite(actual).all().item()
            and torch.isfinite(reference).all().item()
        ),
        **({"expert_coverage": coverage} if coverage is not None else {}),
    }


def _record(
    records: dict,
    name: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    budget: float,
    **extra,
) -> None:
    records[name] = _metrics(actual, reference, budget, **extra)


def _per_expert_route_report(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    if actual.shape != (32, 8, 2048) or tuple(reference.shape) != tuple(actual.shape):
        raise AssertionError("forced route capture must be [32,8,2048]")
    values = [
        _metrics(actual[e // 8, e % 8], reference[e // 8, e % 8], BUDGETS["local"])[
            "nrmse"
        ]
        for e in range(256)
    ]
    finite = bool(
        torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item()
    )
    maximum = max(values) if finite else float("nan")
    return {
        "shape": [256, 2048],
        "nrmse": values,
        "max_expert_nrmse": maximum,
        "finite": finite,
        "budget": BUDGETS["local"],
        "expert_coverage": 256,
        "pass": bool(finite and maximum <= BUDGETS["local"]),
    }


def _production_attention(
    hidden: torch.Tensor, w: dict, positions: torch.Tensor, cu: torch.Tensor
):
    norm = gemma_rms_norm(hidden, w["input_layernorm.weight"])
    qpayload = quantize_fp8(norm)
    merged_qkv = linear_fp8(qpayload, w["self_attn.qkv_proj"])
    qgate, key, value = (
        merged_qkv[:, :8192],
        merged_qkv[:, 8192:8704],
        merged_qkv[:, 8704:],
    )
    q, k, gate = full_qk_rope_gate(
        qgate,
        key,
        positions,
        w["self_attn.q_norm.weight"],
        w["self_attn.k_norm.weight"],
    )
    attended = causal_gqa(q, k, value.reshape(-1, 2, 256), cu, hidden.shape[0])
    out_input, captured_boundary = gated_attention_fp8(
        attended, gate, capture_boundary=True
    )
    out = linear_fp8(out_input, w["self_attn.o_proj"])
    return (
        out,
        norm,
        {
            "self_attn.q_proj": qgate,
            "self_attn.k_proj": key,
            "self_attn.v_proj": value,
            "self_attn.o_proj": out,
        },
        out_input,
        {
            "q": q,
            "k": k,
            "gate": gate,
            "attended": attended,
            "boundary": captured_boundary,
        },
    )


def _production_gdn(hidden: torch.Tensor, w: dict, cu: torch.Tensor):
    norm = gemma_rms_norm(hidden, w["input_layernorm.weight"])
    qpayload = quantize_fp8(norm)
    merged_qkv_z = linear_fp8(qpayload, w["linear_attn.in_proj_qkv_z"])
    qkv, z = merged_qkv_z[:, :8192], merged_qkv_z[:, 8192:]
    merged_ba = linear_fp16(norm, w["linear_attn.in_proj_ba"])
    b, a = merged_ba[:, :32], merged_ba[:, 32:]
    conv = depthwise_conv4_silu(qkv, w["linear_attn.conv1d.weight"], None, cu)
    q, k, v = split_gdn_qkv(conv)
    q, k = l2_normalize_qk(q, k)
    decay, beta = prepare_gates(a, b, w["linear_attn.A_log"], w["linear_attn.dt_bias"])
    value, state = recurrent_gdn(q, k, v, decay, beta, cu, hidden.shape[0])
    out_input, captured_boundary = gated_gdn_fp8(
        value, z, w["linear_attn.norm.weight"], capture_boundary=True
    )
    out = linear_fp8(out_input, w["linear_attn.out_proj"])
    return (
        out,
        norm,
        {
            "linear_attn.in_proj_qkv": qkv,
            "linear_attn.in_proj_z": z,
            "linear_attn.out_proj": out,
        },
        value,
        state,
        out_input,
        {
            "qkv": qkv,
            "z": z,
            "a": a,
            "b": b,
            "conv": conv,
            "q": q,
            "k": k,
            "v": v,
            "decay": decay,
            "beta": beta,
            "boundary": captured_boundary,
        },
    )


def _projection_references(
    norm: torch.Tensor, w: dict, names: tuple[str, ...], output_payload
) -> tuple[dict, dict]:
    """Return explicit mathematical and block-semantic projection oracles."""
    math = {name: reference_model.fp8_linear_math(norm, w[name]) for name in names[:-1]}
    semantic = {name: reference_model.fp8_linear(norm, w[name]) for name in names[:-1]}
    math[names[-1]] = reference_model.fp8_linear_payload_math(
        output_payload.data, output_payload.block_scale, w[names[-1]]
    )
    semantic[names[-1]] = reference_model.fp8_linear_payload(
        output_payload.data, output_payload.block_scale, w[names[-1]]
    )
    return math, semantic


def _reference_moe_fixed_ids(
    x: torch.Tensor, w: dict, ids: torch.Tensor, route_weights: torch.Tensor
) -> torch.Tensor:
    gate_up, down = w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    if gate_up.kind == "fp16":
        routed = reference_moe.fp16_routed_experts(x, gate_up, down, ids, route_weights)
    else:
        routed = reference_moe.routed_experts(
            x,
            gate_up,
            down,
            ids,
            route_weights,
            quantize_a4=reference_model.quantize_a4_rne,
        )
    shared = reference_moe.shared_expert(
        x,
        w["mlp.shared_expert.gate_up_proj"],
        w["mlp.shared_expert.down_proj"],
        w["mlp.shared_expert_gate"],
    )
    return (routed.float() + shared.float()).to(torch.float16)


def _parse_layers(spec: str) -> tuple[int, ...]:
    answer: list[int] = []
    for item in spec.split(","):
        if "-" in item:
            begin, end = (int(x) for x in item.split("-", 1))
            answer.extend(range(begin, end + 1))
        else:
            answer.append(int(item))
    if (
        not answer
        or any(x < 0 or x >= 40 for x in answer)
        or len(set(answer)) != len(answer)
    ):
        raise ValueError("layers must be unique IDs in [0,39]")
    return tuple(answer)


def scan_layer(
    layer_id: int,
    *,
    model_dir: Path = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT,
    tokens: int = 32,
) -> dict:
    """Scan one layer and write its component-level JSON record."""
    if torch.cuda.get_device_capability() != (7, 0):
        raise AssertionError("M3 scan requires SM70")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED + layer_id)
    hidden = (
        torch.randn((tokens, 2048), device="cuda", dtype=torch.float16) * 0.1
    ).contiguous()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    # This is deliberately inside this routine; no multi-layer runner retains
    # previous layer allocations.
    weights = Qwen35Checkpoint(model_dir).load_layer(layer_id, device="cuda")
    records: dict[str, dict] = {}
    try:
        if "linear_attn.in_proj_qkv" in weights:
            (
                projected,
                norm,
                actual_projections,
                state_value,
                state,
                output_payload,
                gdn_actual,
            ) = _production_gdn(hidden, weights, cu)
            names = (
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.out_proj",
            )
        else:
            projected, norm, actual_projections, output_payload, attention_parts = (
                _production_attention(hidden, weights, positions, cu)
            )
            names = (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            )
            state_value = state = None
        residual, post = residual_add_gemma_rms_norm(
            hidden, projected, weights["post_attention_layernorm.weight"]
        )
        reference_final, reference_parts = reference_model.layer(
            hidden, weights, positions
        )
        _record(
            records,
            "input_norm",
            norm,
            reference_model.gemma_rms_norm(hidden, weights["input_layernorm.weight"]),
            BUDGETS["local"],
        )
        projection_math, projection_semantic = _projection_references(
            norm, weights, names, output_payload
        )
        for name in names:
            _record(
                records,
                f"{name}_math",
                actual_projections[name],
                projection_math[name],
                BUDGETS["projection"],
            )
            _record(
                records,
                f"{name}_semantic",
                actual_projections[name],
                projection_semantic[name],
                BUDGETS["projection"],
            )
        _record(
            records,
            "attention_or_gdn",
            projected,
            reference_parts["projected"],
            BUDGETS["local"],
        )
        _record(
            records,
            "post_attention_norm",
            post,
            reference_parts["post_norm"],
            BUDGETS["local"],
        )
        _record(
            records, "residual", residual, reference_parts["residual"], BUDGETS["local"]
        )
        if "linear_attn.in_proj_qkv" in weights:
            for name in ("a", "b", "conv", "q", "k", "v", "decay", "beta"):
                _record(
                    records,
                    f"gdn_{name}",
                    gdn_actual[name],
                    reference_parts[name],
                    BUDGETS["projection"] if name in ("a", "b") else BUDGETS["local"],
                )
            value16 = state_value.to(torch.float16).float()
            gdn_boundary_math = (
                (
                    value16
                    * torch.rsqrt(value16.square().mean(-1, keepdim=True) + 1e-6)
                    * weights["linear_attn.norm.weight"].float()[None, None, :]
                    * torch.nn.functional.silu(gdn_actual["z"].float()).reshape(
                        tokens, 32, 128
                    )
                )
                .to(torch.float16)
                .reshape(tokens, 4096)
            )
            _record(
                records,
                "gdn_a8_boundary",
                gdn_actual["boundary"],
                gdn_boundary_math,
                BUDGETS["projection"],
            )
            gdn_data, gdn_scale = reference_model.quantize_a8_rne(
                gdn_actual["boundary"]
            )
            records["gdn_a8_codec"] = {
                "shape": list(gdn_actual["boundary"].shape),
                "payload_bytes_exact": bool(
                    torch.equal(output_payload.data.cpu(), gdn_data)
                ),
                "scale_exact": bool(
                    torch.equal(output_payload.block_scale.cpu(), gdn_scale)
                ),
                "finite": True,
                "budget": "exact codec bytes/scales",
                "pass": bool(
                    torch.equal(output_payload.data.cpu(), gdn_data)
                    and torch.equal(output_payload.block_scale.cpu(), gdn_scale)
                ),
            }
        if "linear_attn.in_proj_qkv" not in weights:
            _record(
                records,
                "full_q_rope",
                attention_parts["q"],
                reference_parts["q"],
                BUDGETS["local"],
            )
            _record(
                records,
                "full_k_rope",
                attention_parts["k"],
                reference_parts["k"],
                BUDGETS["local"],
            )
            _record(
                records,
                "full_attention",
                attention_parts["attended"],
                reference_parts["attended"],
                BUDGETS["local"],
            )
            actual_gate_boundary = (
                (
                    attention_parts["attended"].float()
                    * torch.sigmoid(attention_parts["gate"].float())
                )
                .to(torch.float16)
                .reshape(tokens, -1)
            )
            _record(
                records,
                "full_gate_boundary",
                actual_gate_boundary,
                reference_parts["gated"],
                BUDGETS["local"],
            )
            # Codec bytes are checked on the identical gate-boundary operand;
            # full_attention above separately checks that independently
            # calculated operand.
            _record(
                records,
                "full_a8_boundary",
                attention_parts["boundary"],
                actual_gate_boundary,
                BUDGETS["projection"],
            )
            payload_data, payload_scale = reference_model.quantize_a8_rne(
                attention_parts["boundary"]
            )
            payload_exact = bool(
                torch.equal(output_payload.data.cpu(), payload_data)
                and torch.equal(output_payload.block_scale.cpu(), payload_scale)
            )
            old_byte_diff = int(
                (output_payload.data.cpu() != reference_parts["a8_data"]).sum().item()
            )
            records["full_a8_codec"] = {
                "shape": list(output_payload.data.shape),
                "payload_bytes_exact": payload_exact,
                "scale_exact": bool(
                    torch.equal(output_payload.block_scale.cpu(), payload_scale)
                ),
                "finite": True,
                "budget": "exact codec bytes/scales",
                "pass": payload_exact,
            }
            records["full_a8_independent_input_diagnostic"] = {
                "shape": list(output_payload.data.shape),
                "payload_byte_differences": old_byte_diff,
                "status": "reported; independent attention input differs before producer",
            }
            _record(
                records,
                "full_o_semantic",
                projected,
                reference_parts["projected"],
                BUDGETS["local"],
            )

        moe_weights = MoeWeights(
            router=weights["mlp.gate"],
            gate_up=weights["mlp.experts.gate_up_proj"],
            down=weights["mlp.experts.down_proj"],
            shared_gate_up=weights["mlp.shared_expert.gate_up_proj"],
            shared_down=weights["mlp.shared_expert.down_proj"],
            shared_gate=weights["mlp.shared_expert_gate"],
        )
        observed_moe = fused_moe(post, moe_weights)
        expected_moe, _, _, _ = reference_moe.moe(
            post,
            weights["mlp.gate"],
            weights["mlp.experts.gate_up_proj"],
            weights["mlp.experts.down_proj"],
            weights["mlp.shared_expert.gate_up_proj"],
            weights["mlp.shared_expert.down_proj"],
            weights["mlp.shared_expert_gate"],
            quantize_a4=reference_model.quantize_a4_rne,
        )
        _record(records, "moe_natural", observed_moe, expected_moe, BUDGETS["local"])
        actual_route_logits = linear_fp16(post, weights["mlp.gate"])
        router_reference = (post.float() @ weights["mlp.gate"].data.float().t()).to(
            torch.float16
        )
        _record(
            records,
            "router_projection",
            actual_route_logits,
            router_reference,
            BUDGETS["projection"],
        )
        actual_route_ids, actual_route_weights = route_topk(actual_route_logits)
        reference_route_ids = reference_parts["route_ids"]
        reference_route_logits = reference_parts["router_logits"]
        actual_sorted = torch.sort(
            actual_route_logits.float(), dim=-1, descending=True
        ).values
        reference_sorted = torch.sort(
            reference_route_logits.float(), dim=-1, descending=True
        ).values
        records["composed_route_ids"] = {
            "shape": list(actual_route_ids.shape),
            "ids_exact": bool(torch.equal(actual_route_ids, reference_route_ids)),
            "actual_margin_8_9_min": (actual_sorted[:, 7] - actual_sorted[:, 8])
            .min()
            .item(),
            "reference_margin_8_9_min": (
                reference_sorted[:, 7] - reference_sorted[:, 8]
            )
            .min()
            .item(),
            "finite": bool(
                torch.isfinite(actual_route_logits).all().item()
                and torch.isfinite(reference_route_logits).all().item()
            ),
            "budget": "diagnostic: propagated post-norm router IDs",
        }
        ref_post = reference_parts["post_norm"]
        ref_logits = (ref_post.float() @ weights["mlp.gate"].data.float().t()).to(
            torch.float16
        )
        ref_prob = torch.softmax(ref_logits.float(), dim=-1)
        ref_same_weights = ref_prob.gather(-1, actual_route_ids.long())
        ref_same_weights = ref_same_weights / ref_same_weights.sum(-1, keepdim=True)
        ref_post_same_ids = _reference_moe_fixed_ids(
            ref_post, weights, actual_route_ids, ref_same_weights
        )
        post_effect = _metrics(expected_moe, ref_post_same_ids, BUDGETS["local"])
        post_effect.pop("pass")
        post_effect.update(
            {
                "budget": None,
                "status": "reported; propagated post-norm difference with identical IDs",
            }
        )
        records["moe_propagated_post_same_ids"] = post_effect
        route_effect = _metrics(
            reference_parts["moe"], ref_post_same_ids, BUDGETS["local"]
        )
        route_effect.pop("pass")
        route_effect.update(
            {"budget": None, "status": "reported; natural vs same propagated route IDs"}
        )
        records["moe_propagated_route_effect"] = route_effect
        observed_final = (residual.float() + observed_moe.float()).to(torch.float16)
        composed = _metrics(observed_final, reference_final, BUDGETS["local"])
        composed.pop("pass")
        composed["budget"] = None
        composed["status"] = "reported; no whole-layer acceptance budget is assigned"
        records["layer_output_composed"] = composed
        _record(
            records,
            "layer_output_local",
            observed_final,
            (residual.float() + expected_moe.float()).to(torch.float16),
            BUDGETS["local"],
        )

        forced_ids = torch.arange(256, dtype=torch.int32, device="cuda").reshape(32, 8)
        forced_weights = torch.full((32, 8), 1 / 8, dtype=torch.float32, device="cuda")
        forced_x = post[:32].contiguous()
        if weights["mlp.experts.gate_up_proj"].kind == "fp16":
            forced_actual, captured_routes, inverse = execute_fp16_experts(
                forced_x,
                weights["mlp.experts.gate_up_proj"],
                weights["mlp.experts.down_proj"],
                forced_ids,
                forced_weights,
                capture_routes=True,
            )
            forced_ref, reference_routes = reference_moe.fp16_routed_experts(
                forced_x,
                weights["mlp.experts.gate_up_proj"],
                weights["mlp.experts.down_proj"],
                forced_ids,
                forced_weights,
                return_routes=True,
            )
        else:
            forced_actual, captured_routes, inverse = execute_experts(
                forced_x,
                weights["mlp.experts.gate_up_proj"],
                weights["mlp.experts.down_proj"],
                forced_ids,
                forced_weights,
                capture_routes=True,
            )
            forced_ref, reference_routes = reference_moe.routed_experts(
                forced_x,
                weights["mlp.experts.gate_up_proj"],
                weights["mlp.experts.down_proj"],
                forced_ids,
                forced_weights,
                quantize_a4=reference_model.quantize_a4_rne,
                return_routes=True,
            )
        _record(
            records,
            "routed_forced_all_256",
            forced_actual,
            forced_ref,
            BUDGETS["local"],
            coverage=256,
        )
        actual_routes = captured_routes[inverse.long()].reshape(32, 8, 2048)
        records["routed_forced_per_expert"] = _per_expert_route_report(
            actual_routes, reference_routes
        )
        shared_gu_actual = linear_fp16(
            forced_x, weights["mlp.shared_expert.gate_up_proj"]
        )
        shared_z_actual = _fp16_swiglu_values(shared_gu_actual)
        shared_value_actual = linear_fp16(
            shared_z_actual, weights["mlp.shared_expert.down_proj"]
        )
        shared_actual = shared_value_actual
        shared_gate = linear_fp16(forced_x, weights["mlp.shared_expert_gate"])
        shared_actual = _fp16_sigmoid_multiply(shared_actual, shared_gate)
        shared_gu_ref = (
            forced_x.float()
            @ weights["mlp.shared_expert.gate_up_proj"].data.float().t()
        ).to(torch.float16)
        shared_z_ref = (
            torch.nn.functional.silu(shared_gu_ref[:, :512].float())
            * shared_gu_ref[:, 512:].float()
        ).to(torch.float16)
        shared_value_ref = (
            shared_z_ref.float()
            @ weights["mlp.shared_expert.down_proj"].data.float().t()
        ).to(torch.float16)
        shared_gate_ref = (
            forced_x.float() @ weights["mlp.shared_expert_gate"].data.float().t()
        ).to(torch.float16)
        _record(
            records,
            "shared_gate_up_projection",
            shared_gu_actual,
            shared_gu_ref,
            BUDGETS["projection"],
        )
        _record(
            records, "shared_swiglu", shared_z_actual, shared_z_ref, BUDGETS["local"]
        )
        _record(
            records,
            "shared_down_projection",
            shared_value_actual,
            shared_value_ref,
            BUDGETS["projection"],
        )
        _record(
            records,
            "shared_gate_projection",
            shared_gate,
            shared_gate_ref,
            BUDGETS["projection"],
        )
        shared_ref = reference_moe.shared_expert(
            forced_x,
            weights["mlp.shared_expert.gate_up_proj"],
            weights["mlp.shared_expert.down_proj"],
            weights["mlp.shared_expert_gate"],
        )
        _record(records, "shared_expert", shared_actual, shared_ref, BUDGETS["local"])

        # T17 is intentionally a separate natural-router sample.  Same-logit
        # ties must select lower expert IDs; a genuine FP16-matmul swap is
        # recorded with the 8/9 score margin and never changes a tolerance.
        torch.manual_seed(SEED + 10_000 + layer_id)
        natural = (
            torch.randn((17, 2048), device="cuda", dtype=torch.float16) * 0.1
        ).contiguous()
        router_logits = linear_fp16(natural, weights["mlp.gate"])
        prod_ids, _ = route_topk(router_logits)
        same_logit_ids, _ = reference_moe.stable_top8(router_logits)
        if not torch.equal(prod_ids, same_logit_ids):
            raise AssertionError(
                "same-logit stable Top-8 must select identical lower-ID ties"
            )
        ref_ids, _ = reference_moe.stable_top8(
            (natural.float() @ weights["mlp.gate"].data.float().t()).to(torch.float16)
        )
        sorted_scores = torch.sort(
            router_logits.float(), dim=-1, descending=True
        ).values
        records["router_natural_t17"] = {
            "shape": [17, 8],
            "same_logits_ids_exact": True,
            "fp16_matmul_ids_exact": bool(torch.equal(prod_ids, ref_ids)),
            "score_margin_8_9_min": (sorted_scores[:, 7] - sorted_scores[:, 8])
            .min()
            .item(),
            "finite": bool(torch.isfinite(router_logits).all().item()),
            "budget": "exact lower-ID IDs for same logits; matmul swaps reported",
        }
        ties = torch.zeros((1, 256), device="cuda", dtype=torch.float16)
        tied_ids, _ = route_topk(ties)
        if tied_ids[0].tolist() != list(range(8)):
            raise AssertionError("stable router ties must choose lower expert IDs")
        if state_value is not None:
            same_value, same_state = reference_recurrent(
                gdn_actual["q"],
                gdn_actual["k"],
                gdn_actual["v"],
                gdn_actual["decay"],
                gdn_actual["beta"],
                cu,
            )
            _record(
                records,
                "gdn_recurrent_output",
                state_value,
                same_value,
                BUDGETS["gdn_recurrent"],
            )
            _record(
                records,
                "gdn_recurrent_state",
                state,
                same_state,
                BUDGETS["gdn_recurrent"],
            )
            propagated_value = _metrics(
                state_value, reference_parts["gdn_value"], BUDGETS["gdn_recurrent"]
            )
            propagated_value.pop("pass")
            propagated_value.update(
                {
                    "budget": None,
                    "status": "reported; independently propagated GDN inputs",
                }
            )
            records["gdn_recurrent_propagated_output"] = propagated_value
        payload = {
            "layer": layer_id,
            "tokens": tokens,
            "seed": SEED + layer_id,
            "math_contract": MATH_CONTRACT,
            "expert_execution_backend": "fused paired G1/SwiGLU/A4",
            "source_sha256": _RUN_SOURCE_HASH or _source_hash(),
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "uuid_required": V100_UUID,
                "platform": platform.platform(),
            },
            "budgets": BUDGETS,
            "components": records,
        }
        path = output_dir / f"layer_{layer_id:02d}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        failures = [
            name for name, record in records.items() if record.get("pass") is False
        ]
        if failures:
            raise AssertionError(
                f"layer {layer_id} acceptance failures: {failures}; see {path}"
            )
        return payload
    finally:
        del weights
        torch.cuda.empty_cache()


def scan_layers(
    layer_ids: tuple[int, ...],
    *,
    model_dir: Path = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT,
) -> list[dict]:
    output = []
    for layer in layer_ids:
        print(f"M3 scan: layer {layer}", flush=True)
        output.append(scan_layer(layer, model_dir=model_dir, output_dir=output_dir))
    return output


class TestLayerScanSmoke(V100TestCase):
    @unittest.skipUnless(
        os.environ.get("QWEN35_LAYER_SCAN_SMOKE") == "1",
        "set QWEN35_LAYER_SCAN_SMOKE=1 to run real-checkpoint scan smoke",
    )
    @unittest.skipUnless(DEFAULT_MODEL.is_dir(), "real Qwen3.5 checkpoint unavailable")
    def test_one_real_layer(self):
        report = scan_layer(0)
        self.assertEqual(report["layer"], 0)


def main(argv: list[str] | None = None) -> int:
    global _RUN_SOURCE_HASH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="0-39")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.model_dir.is_dir():
        parser.error(f"checkpoint unavailable: {args.model_dir}")
    if V100_UUID not in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        parser.error("set CUDA_VISIBLE_DEVICES to the required V100 UUID")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        parser.error("M3 scan requires an available SM70 V100")
    _RUN_SOURCE_HASH = _source_hash()
    # The standalone CLI owns serialization.  The unittest helper calls
    # scan_layer directly and V100TestCase already holds this same lock.
    lock = open("/tmp/qwen35-v100-gpu.lock", "a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        result = scan_layers(
            _parse_layers(args.layers),
            model_dir=args.model_dir,
            output_dir=args.output_dir,
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    print(
        json.dumps(
            {
                "layers": [item["layer"] for item in result],
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
