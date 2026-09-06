"""Generated exact safetensors header contract for the supported Qwen3.5 file."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeaderSpec:
    shape: tuple[int, ...]
    dtype: str


P = "model.language_model."


def _fp8(n: int, k: int) -> dict[str, HeaderSpec]:
    return {
        "weight": HeaderSpec((n, k), "F8_E4M3"),
        "weight_scale": HeaderSpec((n // 128, k // 128), "F16"),
    }


def _expert(layer: int, expert: int, projection: str) -> dict[str, HeaderSpec]:
    n, k = (512, 2048) if projection in ("gate_proj", "up_proj") else (2048, 512)
    stem = f"{P}layers.{layer}.mlp.experts.{expert}.{projection}."
    if layer in (0, 39):
        return {stem + "weight": HeaderSpec((n, k), "F16")}
    return {
        stem + "weight_packed": HeaderSpec((n, k // 2), "U8"),
        stem + "weight_scale": HeaderSpec((n, k // 16), "F8_E4M3"),
        stem + "weight_global_scale": HeaderSpec((1,), "F32"),
        stem + "input_global_scale": HeaderSpec((1,), "F32"),
    }


def expected_headers() -> dict[str, HeaderSpec]:
    """Build all and only text/checkpoint headers used by the compat runner."""
    out = {
        f"{P}embed_tokens.weight": HeaderSpec((248320, 2048), "F16"),
        f"{P}norm.weight": HeaderSpec((2048,), "F16"),
        "lm_head.weight": HeaderSpec((248320, 2048), "F16"),
    }
    common = {
        "input_layernorm.weight": HeaderSpec((2048,), "F16"),
        "post_attention_layernorm.weight": HeaderSpec((2048,), "F16"),
        "mlp.gate.weight": HeaderSpec((256, 2048), "F16"),
        "mlp.shared_expert.gate_proj.weight": HeaderSpec((512, 2048), "F16"),
        "mlp.shared_expert.up_proj.weight": HeaderSpec((512, 2048), "F16"),
        "mlp.shared_expert.down_proj.weight": HeaderSpec((2048, 512), "F16"),
        "mlp.shared_expert_gate.weight": HeaderSpec((1, 2048), "F16"),
    }
    gdn = {
        "linear_attn.A_log": HeaderSpec((32,), "F16"),
        "linear_attn.dt_bias": HeaderSpec((32,), "F16"),
        "linear_attn.conv1d.weight": HeaderSpec((8192, 1, 4), "F16"),
        "linear_attn.in_proj_a.weight": HeaderSpec((32, 2048), "F16"),
        "linear_attn.in_proj_b.weight": HeaderSpec((32, 2048), "F16"),
        "linear_attn.norm.weight": HeaderSpec((128,), "F16"),
    }
    for name, (n, k) in {
        "in_proj_qkv": (8192, 2048),
        "in_proj_z": (4096, 2048),
        "out_proj": (2048, 4096),
    }.items():
        for suffix, spec in _fp8(n, k).items():
            gdn[f"linear_attn.{name}.{suffix}"] = spec
    full = {
        "self_attn.q_norm.weight": HeaderSpec((256,), "F16"),
        "self_attn.k_norm.weight": HeaderSpec((256,), "F16"),
    }
    for name, (n, k) in {
        "q_proj": (8192, 2048),
        "k_proj": (512, 2048),
        "v_proj": (512, 2048),
        "o_proj": (2048, 4096),
    }.items():
        for suffix, spec in _fp8(n, k).items():
            full[f"self_attn.{name}.{suffix}"] = spec
    for layer in range(40):
        for relative, spec in (common | (gdn if layer % 4 != 3 else full)).items():
            out[f"{P}layers.{layer}.{relative}"] = spec
        for expert in range(256):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                out.update(_expert(layer, expert, projection))
    return out


EXPECTED_HEADERS = expected_headers()
