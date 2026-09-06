"""Streaming reader and structural audit for Qwen3.5 NVFP4 checkpoints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch
from safetensors import safe_open

from .manifest import EXPECTED_HEADERS, HeaderSpec
from .weights import Weight

MODEL_PREFIX = "model.language_model."
GLOBAL_TENSORS = {
    "embed_tokens": "model.language_model.embed_tokens.weight",
    "final_norm": "model.language_model.norm.weight",
    "lm_head": "lm_head.weight",
}
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.+)$")


@dataclass(frozen=True)
class CheckpointAudit:
    fp8_matrices: int
    nvfp4_matrices: int
    fp16_expert_matrices: int
    layers: int


class Qwen35Checkpoint:
    """Index-backed loader that only materializes one selected layer at a time.

    The loader returns a dictionary keyed by layer-relative module name. Dense
    projection names omit ``.weight``; scalar/vector tensors retain their
    checkpoint relative key. Experts are packed into stable gate/up and down
    keys required by the grouped MoE ABI.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.validate_config(self.model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        with index_path.open() as handle:
            self._weight_map: Mapping[str, str] = json.load(handle)["weight_map"]
        self._by_layer: Dict[int, Dict[str, str]] = {}
        self._headers: Dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        for name, shard in self._weight_map.items():
            match = LAYER_RE.match(name)
            if match:
                self._by_layer.setdefault(int(match.group(1)), {})[
                    match.group(2)
                ] = shard

    @staticmethod
    def validate_config(model_dir: str | Path) -> Mapping[str, object]:
        """Validate only the fixed text-only shape contract before loading GBs.

        This intentionally does not instantiate a Transformers config: 4.43
        resolves unknown ``qwen3_5_moe`` model types before SGLang's old
        registry has a chance to intervene.
        """
        path = Path(model_dir) / "config.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open() as handle:
            config: Mapping[str, object] = json.load(handle)
        if config.get("model_type") != "qwen3_5_moe" or not config.get(
            "language_model_only", False
        ):
            raise ValueError(
                "only language_model_only qwen3_5_moe checkpoints are supported"
            )
        text = config.get("text_config")
        if not isinstance(text, Mapping):
            raise ValueError("qwen3_5_moe config must contain text_config")
        expected = {
            "vocab_size": 248320,
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "shared_expert_intermediate_size": 512,
            "moe_intermediate_size": 512,
            "rms_norm_eps": 1e-6,
        }
        for key, value in expected.items():
            if text.get(key) != value:
                raise ValueError(
                    f"unsupported Qwen3.5 text_config {key}={text.get(key)!r}"
                )
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        theta = (
            rope.get("rope_theta", rope.get("theta", 10_000_000.0))
            if isinstance(rope, Mapping)
            else None
        )
        if theta != 10_000_000.0:
            raise ValueError(f"unsupported Qwen3.5 RoPE theta={theta!r}")
        required = {
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "attn_output_gate": True,
            "output_gate_type": "swish",
            "hidden_act": "silu",
            "attention_bias": False,
        }
        for key, value in required.items():
            if text.get(key) != value:
                raise ValueError(
                    f"unsupported Qwen3.5 text_config {key}={text.get(key)!r}"
                )
        if config.get("tie_word_embeddings", False) or text.get(
            "tie_word_embeddings", False
        ):
            raise ValueError("tied embeddings are unsupported")
        if (
            not isinstance(rope, Mapping)
            or rope.get("rope_type") != "default"
            or rope.get("partial_rotary_factor") != 0.25
        ):
            raise ValueError("expected default partial-0.25 RoPE")
        layers = text.get("layer_types")
        if (
            layers
            != [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            * 10
        ):
            raise ValueError("expected the fixed [GDN,GDN,GDN,Full] * 10 layer layout")
        return config

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_layer))

    def _shape_dtype(self, name: str) -> tuple[tuple[int, ...], object]:
        cached = self._headers.get(name)
        if cached is not None:
            return cached
        shard = self._weight_map[name]
        with safe_open(
            str(self.model_dir / shard), framework="pt", device="cpu"
        ) as reader:
            piece = reader.get_slice(name)
            value = (tuple(piece.get_shape()), piece.get_dtype())
        self._headers[name] = value
        return value

    def _populate_headers(self) -> None:
        """Read safetensors headers once per shard, never 58k file opens."""
        by_shard: Dict[str, list[str]] = {}
        for name, shard in self._weight_map.items():
            if name not in self._headers:
                by_shard.setdefault(shard, []).append(name)
        for shard, names in by_shard.items():
            with safe_open(
                str(self.model_dir / shard), framework="pt", device="cpu"
            ) as reader:
                for name in names:
                    piece = reader.get_slice(name)
                    self._headers[name] = (tuple(piece.get_shape()), piece.get_dtype())

    @staticmethod
    def _dtype_name(dtype: object) -> str:
        if isinstance(dtype, str):
            return dtype
        return {
            torch.float16: "F16",
            torch.float32: "F32",
            torch.float8_e4m3fn: "F8_E4M3",
            torch.uint8: "U8",
        }.get(dtype, str(dtype))

    def _validate_headers(self, names: Iterable[str]) -> None:
        for name in names:
            spec = EXPECTED_HEADERS.get(name)
            if spec is None:
                raise ValueError(f"unexpected text checkpoint tensor: {name}")
            shape, dtype = self._shape_dtype(name)
            if shape != spec.shape or self._dtype_name(dtype) != spec.dtype:
                raise ValueError(
                    f"invalid header for {name}: {(shape, self._dtype_name(dtype))}, expected {(spec.shape, spec.dtype)}"
                )

    def audit(self) -> CheckpointAudit:
        """Check all quantized matrix companions without loading their payload."""
        self._populate_headers()
        actual = {
            name
            for name in self._weight_map
            if name.startswith(MODEL_PREFIX) or name == "lm_head.weight"
        }
        expected = set(EXPECTED_HEADERS)
        if actual != expected:
            missing, extra = expected - actual, actual - expected
            raise ValueError(
                f"text checkpoint header names differ; missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
            )
        self._validate_headers(expected)
        if self.layer_ids != tuple(range(40)):
            raise ValueError(f"expected layers 0..39, got {self.layer_ids}")
        fp8 = nvfp4 = fp16_experts = 0
        for layer, tensors in self._by_layer.items():
            for relative in tensors:
                if relative.endswith(".weight_packed"):
                    base = relative[: -len(".weight_packed")]
                    required = [
                        base + ".weight_scale",
                        base + ".weight_global_scale",
                        base + ".input_global_scale",
                    ]
                    if any(key not in tensors for key in required):
                        raise ValueError(
                            f"missing NVFP4 companion for layer {layer}: {base}"
                        )
                    if layer in (0, 39):
                        raise ValueError(
                            f"layer {layer} routed experts must be FP16: {base}"
                        )
                    packed_shape, packed_dtype = self._shape_dtype(
                        f"{MODEL_PREFIX}layers.{layer}.{relative}"
                    )
                    scale_shape, scale_dtype = self._shape_dtype(
                        f"{MODEL_PREFIX}layers.{layer}.{base}.weight_scale"
                    )
                    global_shape, global_dtype = self._shape_dtype(
                        f"{MODEL_PREFIX}layers.{layer}.{base}.weight_global_scale"
                    )
                    input_shape, input_dtype = self._shape_dtype(
                        f"{MODEL_PREFIX}layers.{layer}.{base}.input_global_scale"
                    )
                    if packed_dtype not in (torch.uint8, "U8") or scale_dtype not in (
                        torch.float8_e4m3fn,
                        "F8_E4M3",
                    ):
                        raise TypeError(
                            f"invalid NVFP4 storage for layer {layer}: {base}"
                        )
                    if len(packed_shape) != 2 or scale_shape != (
                        packed_shape[0],
                        packed_shape[1] // 8,
                    ):
                        raise ValueError(
                            f"invalid NVFP4 scale shape for layer {layer}: {base}"
                        )
                    if (
                        global_shape != (1,)
                        or input_shape != (1,)
                        or global_dtype not in (torch.float32, "F32")
                        or input_dtype not in (torch.float32, "F32")
                    ):
                        raise ValueError(
                            f"invalid NVFP4 global scale for layer {layer}: {base}"
                        )
                    nvfp4 += 1
                elif relative.endswith(".weight") and relative + "_scale" in tensors:
                    shape, dtype = self._shape_dtype(
                        f"{MODEL_PREFIX}layers.{layer}.{relative}"
                    )
                    scale_shape, scale_dtype = self._shape_dtype(
                        f"{MODEL_PREFIX}layers.{layer}.{relative}_scale"
                    )
                    if dtype not in (
                        torch.float8_e4m3fn,
                        "F8_E4M3",
                    ) or scale_dtype not in (torch.float16, "F16"):
                        raise TypeError(
                            f"invalid FP8 storage for layer {layer}: {relative}"
                        )
                    if len(shape) != 2 or scale_shape != (
                        (shape[0] + 127) // 128,
                        (shape[1] + 127) // 128,
                    ):
                        raise ValueError(
                            f"invalid FP8 scale shape for layer {layer}: {relative}"
                        )
                    fp8 += 1
                elif (
                    layer in (0, 39)
                    and EXPERT_RE.match(relative)
                    and relative.endswith(".weight")
                ):
                    shape, dtype = self._shape_dtype(
                        f"{MODEL_PREFIX}layers.{layer}.{relative}"
                    )
                    if len(shape) != 2 or dtype not in (torch.float16, "F16"):
                        raise TypeError(
                            f"layer {layer} FP16 expert is malformed: {relative}"
                        )
                    fp16_experts += 1
        if fp8 != 130:
            raise ValueError(f"expected 130 FP8 matrices, found {fp8}")
        if nvfp4 != 29184:
            raise ValueError(f"expected 29184 NVFP4 matrices, found {nvfp4}")
        if fp16_experts != 1536:
            raise ValueError(
                f"expected 1536 FP16 routed expert matrices, found {fp16_experts}"
            )
        return CheckpointAudit(fp8, nvfp4, fp16_experts, len(self.layer_ids))

    def _read_layer(
        self, layer: int, device: Optional[torch.device | str]
    ) -> Dict[str, torch.Tensor]:
        if layer not in self._by_layer:
            raise KeyError(f"unknown layer {layer}")
        by_shard: Dict[str, list[str]] = {}
        for relative, shard in self._by_layer[layer].items():
            by_shard.setdefault(shard, []).append(relative)
        result: Dict[str, torch.Tensor] = {}
        for shard, relatives in by_shard.items():
            with safe_open(
                str(self.model_dir / shard), framework="pt", device="cpu"
            ) as reader:
                for relative in relatives:
                    tensor = reader.get_tensor(
                        f"{MODEL_PREFIX}layers.{layer}.{relative}"
                    )
                    if device is not None:
                        tensor = tensor.to(device, non_blocking=True)
                    result[relative] = tensor
        return result

    @staticmethod
    def _raw_fp8(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.float8_e4m3fn:
            raise TypeError(f"expected E4M3FN tensor, found {tensor.dtype}")
        return tensor.view(torch.uint8)

    @staticmethod
    def _stack_experts(
        raw: Mapping[str, torch.Tensor], projected: str, kind: str
    ) -> Weight:
        values = []
        scales = []
        globals_ = []
        inputs = []
        for expert in range(256):
            stem = f"mlp.experts.{expert}.{projected}."
            if kind == "fp16":
                values.append(raw[stem + "weight"])
            else:
                values.append(raw[stem + "weight_packed"])
                scales.append(Qwen35Checkpoint._raw_fp8(raw[stem + "weight_scale"]))
                globals_.append(raw[stem + "weight_global_scale"].reshape(()))
                inputs.append(raw[stem + "input_global_scale"].reshape(()))
        if kind == "fp16":
            data = torch.stack(values)
            return Weight("fp16", data, tuple(data.shape))
        data = torch.stack(values)
        local = torch.stack(scales)
        global_values = torch.stack(globals_)
        input_values = torch.stack(inputs)
        if not bool(
            torch.isfinite(global_values).all()
            and torch.isfinite(input_values).all()
            and (global_values > 0).all()
            and (input_values > 0).all()
        ):
            raise ValueError(
                f"expert {projected} global scales must be finite and positive"
            )
        n, packed_k = data.shape[-2:]
        return Weight(
            "nvfp4",
            data,
            (256, n, packed_k * 2),
            local,
            global_values,
            input_values,
        )

    @staticmethod
    def _pack_experts(raw: Dict[str, torch.Tensor], layer: int) -> Dict[str, Weight]:
        kind = "fp16" if layer in (0, 39) else "nvfp4"
        gate = Qwen35Checkpoint._stack_experts(raw, "gate_proj", kind)
        up = Qwen35Checkpoint._stack_experts(raw, "up_proj", kind)
        down = Qwen35Checkpoint._stack_experts(raw, "down_proj", kind)
        if kind == "fp16":
            gate_up = Weight(
                "fp16",
                torch.cat((gate.data, up.data), dim=1),
                (256, gate.n + up.n, gate.k),
            )
        else:
            if not torch.equal(gate.input_global_scale, up.input_global_scale):
                raise ValueError(f"layer {layer}: gate/up input global scales differ")
            if not torch.equal(gate.global_scale, up.global_scale):
                raise ValueError(f"layer {layer}: gate/up weight global scales differ")
            if not torch.equal(
                gate.input_global_scale,
                gate.input_global_scale[:1].expand_as(gate.input_global_scale),
            ):
                raise ValueError(
                    f"layer {layer}: gate/up input global scale must be static across experts"
                )
            gate_up = Weight(
                "nvfp4",
                torch.cat((gate.data, up.data), dim=1),
                (256, gate.n + up.n, gate.k),
                torch.cat((gate.block_scale, up.block_scale), dim=1),
                gate.global_scale,
                gate.input_global_scale,
            )
        return {"mlp.experts.gate_up_proj": gate_up, "mlp.experts.down_proj": down}

    @staticmethod
    def _pack_shared(raw: Dict[str, torch.Tensor]) -> Dict[str, Weight]:
        gate = raw.pop("mlp.shared_expert.gate_proj.weight")
        up = raw.pop("mlp.shared_expert.up_proj.weight")
        down = raw.pop("mlp.shared_expert.down_proj.weight")
        return {
            "mlp.shared_expert.gate_up_proj": Weight(
                "fp16",
                torch.cat((gate, up), dim=0),
                (gate.shape[0] + up.shape[0], gate.shape[1]),
            ),
            "mlp.shared_expert.down_proj": Weight("fp16", down, tuple(down.shape)),
        }

    @staticmethod
    def _fp8_row_view(merged: Weight, start: int, rows: int) -> Weight:
        """Make a byte/scale row view of a packed FP8 projection."""
        if merged.kind != "fp8" or merged.block_scale is None:
            raise TypeError("expected a packed FP8 projection")
        if start < 0 or rows <= 0 or start + rows > merged.n or start % 128:
            raise ValueError("invalid packed FP8 projection row range")
        return Weight(
            "fp8",
            merged.data.narrow(0, start, rows),
            (rows, merged.k),
            merged.block_scale.narrow(0, start // 128, (rows + 127) // 128),
        )

    @staticmethod
    def _fp16_row_view(merged: Weight, start: int, rows: int) -> Weight:
        """Make a row view of a packed FP16 projection."""
        if merged.kind != "fp16":
            raise TypeError("expected a packed FP16 projection")
        if start < 0 or rows <= 0 or start + rows > merged.n:
            raise ValueError("invalid packed FP16 projection row range")
        return Weight("fp16", merged.data.narrow(0, start, rows), (rows, merged.k))

    @staticmethod
    def _pack_fp8_rows(raw: Dict[str, torch.Tensor], names: tuple[str, ...]) -> Weight:
        """Remove and concatenate checkpoint FP8 rows without dequantizing."""
        values = []
        scales = []
        for name in names:
            value = raw.pop(name + ".weight")
            scale = raw.pop(name + ".weight_scale")
            values.append(Qwen35Checkpoint._raw_fp8(value))
            scales.append(scale)
        k = values[0].shape[1]
        if any(value.ndim != 2 or value.shape[1] != k for value in values):
            raise ValueError("packed FP8 projections must have a shared K")
        return Weight(
            "fp8",
            torch.cat(values, dim=0),
            (sum(value.shape[0] for value in values), k),
            torch.cat(scales, dim=0),
        )

    @staticmethod
    def _pack_gdn_projections(raw: Dict[str, torch.Tensor]) -> Dict[str, Weight]:
        """CPU-pack GDN projections and retain zero-copy component views."""
        qkv_z = Qwen35Checkpoint._pack_fp8_rows(
            raw, ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")
        )
        b = raw.pop("linear_attn.in_proj_b.weight")
        a = raw.pop("linear_attn.in_proj_a.weight")
        if (
            b.dtype != torch.float16
            or a.dtype != torch.float16
            or b.ndim != 2
            or a.shape != b.shape
        ):
            raise ValueError("GDN B/A projections must be matching FP16 matrices")
        ba = Weight(
            "fp16", torch.cat((b, a), dim=0), (b.shape[0] + a.shape[0], b.shape[1])
        )
        return Qwen35Checkpoint._restore_projection_views(
            {
                "linear_attn.in_proj_qkv_z": qkv_z,
                "linear_attn.in_proj_ba": ba,
            }
        )

    @staticmethod
    def _pack_full_projections(raw: Dict[str, torch.Tensor]) -> Dict[str, Weight]:
        """CPU-pack QGate/K/V projections and retain zero-copy component views."""
        qkv = Qwen35Checkpoint._pack_fp8_rows(
            raw, ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        )
        return Qwen35Checkpoint._restore_projection_views({"self_attn.qkv_proj": qkv})

    @staticmethod
    def _restore_projection_views(
        output: Dict[str, torch.Tensor | Weight]
    ) -> Dict[str, torch.Tensor | Weight]:
        """Restore public component keys as row views of packed storage.

        This is also used after the one device transfer.  Constructing these
        views on the destination avoids uploading each alias independently.
        """
        qkv_z = output.get("linear_attn.in_proj_qkv_z")
        if isinstance(qkv_z, Weight):
            qkv_rows = qkv_z.n - 4096
            output["linear_attn.in_proj_qkv"] = Qwen35Checkpoint._fp8_row_view(
                qkv_z, 0, qkv_rows
            )
            output["linear_attn.in_proj_z"] = Qwen35Checkpoint._fp8_row_view(
                qkv_z, qkv_rows, 4096
            )
        ba = output.get("linear_attn.in_proj_ba")
        if isinstance(ba, Weight):
            rows = ba.n // 2
            if ba.n != rows * 2:
                raise ValueError(
                    "GDN B/A packed projection must have an even row count"
                )
            output["linear_attn.in_proj_b"] = Qwen35Checkpoint._fp16_row_view(
                ba, 0, rows
            )
            output["linear_attn.in_proj_a"] = Qwen35Checkpoint._fp16_row_view(
                ba, rows, rows
            )
        qkv = output.get("self_attn.qkv_proj")
        if isinstance(qkv, Weight):
            if qkv.n < 1024:
                raise ValueError(
                    "full-attention packed QGate/K/V projection is too small"
                )
            qgate_rows = qkv.n - 1024
            output["self_attn.q_proj"] = Qwen35Checkpoint._fp8_row_view(
                qkv, 0, qgate_rows
            )
            output["self_attn.k_proj"] = Qwen35Checkpoint._fp8_row_view(
                qkv, qgate_rows, 512
            )
            output["self_attn.v_proj"] = Qwen35Checkpoint._fp8_row_view(
                qkv, qgate_rows + 512, 512
            )
        return output

    @staticmethod
    def _projection_component_keys(
        output: Mapping[str, torch.Tensor | Weight]
    ) -> set[str]:
        keys: set[str] = set()
        if "linear_attn.in_proj_qkv_z" in output:
            keys.update(("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"))
        if "linear_attn.in_proj_ba" in output:
            keys.update(("linear_attn.in_proj_b", "linear_attn.in_proj_a"))
        if "self_attn.qkv_proj" in output:
            keys.update(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))
        return keys

    @staticmethod
    def _move_weight(weight: Weight, device: torch.device | str) -> Weight:
        return Weight(
            weight.kind,
            weight.data.to(device, non_blocking=True),
            weight.logical_shape,
            (
                None
                if weight.block_scale is None
                else weight.block_scale.to(device, non_blocking=True)
            ),
            (
                None
                if weight.global_scale is None
                else weight.global_scale.to(device, non_blocking=True)
            ),
            (
                None
                if weight.input_global_scale is None
                else weight.input_global_scale.to(device, non_blocking=True)
            ),
        )

    def load_layer(
        self, layer: int, device: Optional[torch.device | str] = None
    ) -> Dict[str, torch.Tensor | Weight]:
        """Load one layer, pack on CPU, then make one compact device copy.

        Packing compressed experts directly on GPU creates raw plus stacked plus
        concatenated copies at once.  CPU packing keeps the V100 resident set
        to the final compact layer representation.
        """
        if layer not in self._by_layer:
            raise KeyError(f"unknown layer {layer}")
        actual_names = {
            f"{MODEL_PREFIX}layers.{layer}.{relative}"
            for relative in self._by_layer[layer]
        }
        expected_names = {
            name
            for name in EXPECTED_HEADERS
            if name.startswith(f"{MODEL_PREFIX}layers.{layer}.")
        }
        if actual_names != expected_names:
            raise ValueError(f"layer {layer} header names differ before payload read")
        # Header validation otherwise opens a shard once per tensor, which is
        # especially costly for the thousands of expert tensors in a layer.
        # This reads each safetensors header once, never its payload.
        self._populate_headers()
        self._validate_headers(actual_names)
        raw = self._read_layer(layer, None)
        output: Dict[str, torch.Tensor | Weight] = self._pack_experts(raw, layer)
        output.update(self._pack_shared(raw))
        output.update(
            self._pack_gdn_projections(raw)
            if layer % 4 != 3
            else self._pack_full_projections(raw)
        )
        expert_prefix = "mlp.experts."
        for key in list(raw):
            if key.startswith(expert_prefix):
                raw.pop(key)
        for key, value in raw.items():
            if key.endswith(".weight") and key + "_scale" in raw:
                base = key[: -len(".weight")]
                output[base] = Weight(
                    "fp8", self._raw_fp8(value), tuple(value.shape), raw[key + "_scale"]
                )
            elif key.endswith(".weight_scale"):
                continue
            elif key.endswith(".weight") and value.ndim == 2:
                output[key[: -len(".weight")]] = Weight(
                    "fp16", value, tuple(value.shape)
                )
            elif key == "linear_attn.conv1d.weight" and value.shape[1:2] == (1,):
                # Checkpoint Conv1d storage [C,1,4] is a CPU setup view;
                # runtime receives its direct [C,4] Triton layout.
                output[key] = value[:, 0, :]
            else:
                output[key] = value
        if device is None:
            return output
        # Component projections are storage aliases on CPU.  Only transfer the
        # two/one merged payloads and recreate their views on the target.
        aliases = self._projection_component_keys(output)
        transferred = {
            key: (
                self._move_weight(value, device)
                if isinstance(value, Weight)
                else value.to(device, non_blocking=True)
            )
            for key, value in output.items()
            if key not in aliases
        }
        return self._restore_projection_views(transferred)

    def load_global_tensors(
        self, device: Optional[torch.device | str] = None
    ) -> Dict[str, torch.Tensor | Weight]:
        """Load embedding, final norm and LM head without reading every layer.

        Matrix entries use the same ``Weight`` ABI as layers; final RMS weight
        remains a vector.  Callers can request CUDA after this CPU read, which
        avoids duplicate host/device payloads.
        """
        result: Dict[str, torch.Tensor | Weight] = {}
        self._validate_headers(GLOBAL_TENSORS.values())
        for public, name in GLOBAL_TENSORS.items():
            shard = self._weight_map.get(name)
            if shard is None:
                raise KeyError(f"checkpoint is missing {name}")
            with safe_open(
                str(self.model_dir / shard), framework="pt", device="cpu"
            ) as reader:
                tensor = reader.get_tensor(name)
            if tensor.dtype != torch.float16 or tensor.ndim not in (1, 2):
                raise TypeError(f"{name} must be an FP16 vector or matrix")
            expected_shape = (2048,) if public == "final_norm" else (248320, 2048)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}"
                )
            if tensor.ndim == 2:
                value: torch.Tensor | Weight = Weight(
                    "fp16", tensor, tuple(tensor.shape)
                )
            else:
                value = tensor
            if device is not None:
                value = (
                    self._move_weight(value, device)
                    if isinstance(value, Weight)
                    else value.to(device, non_blocking=True)
                )
            result[public] = value
        return result
