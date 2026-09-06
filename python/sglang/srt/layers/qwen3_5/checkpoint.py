"""Streaming reader and structural audit for Qwen3.5 NVFP4 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Mapping, Optional

import torch
from safetensors import safe_open

from .weights import Weight


MODEL_PREFIX = "model.language_model."
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
                self._by_layer.setdefault(int(match.group(1)), {})[match.group(2)] = shard

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_layer))

    def _shape_dtype(self, name: str) -> tuple[tuple[int, ...], object]:
        cached = self._headers.get(name)
        if cached is not None:
            return cached
        shard = self._weight_map[name]
        with safe_open(str(self.model_dir / shard), framework="pt", device="cpu") as reader:
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
            with safe_open(str(self.model_dir / shard), framework="pt", device="cpu") as reader:
                for name in names:
                    piece = reader.get_slice(name)
                    self._headers[name] = (tuple(piece.get_shape()), piece.get_dtype())

    def audit(self) -> CheckpointAudit:
        """Check all quantized matrix companions without loading their payload."""
        self._populate_headers()
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
                        raise ValueError(f"missing NVFP4 companion for layer {layer}: {base}")
                    if layer in (0, 39):
                        raise ValueError(f"layer {layer} routed experts must be FP16: {base}")
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
                    if packed_dtype not in (torch.uint8, "U8") or scale_dtype not in (torch.float8_e4m3fn, "F8_E4M3"):
                        raise TypeError(f"invalid NVFP4 storage for layer {layer}: {base}")
                    if len(packed_shape) != 2 or scale_shape != (packed_shape[0], packed_shape[1] // 8):
                        raise ValueError(f"invalid NVFP4 scale shape for layer {layer}: {base}")
                    if (global_shape != (1,) or input_shape != (1,) or
                            global_dtype not in (torch.float32, "F32") or
                            input_dtype not in (torch.float32, "F32")):
                        raise ValueError(f"invalid NVFP4 global scale for layer {layer}: {base}")
                    nvfp4 += 1
                elif relative.endswith(".weight") and relative + "_scale" in tensors:
                    shape, dtype = self._shape_dtype(f"{MODEL_PREFIX}layers.{layer}.{relative}")
                    scale_shape, scale_dtype = self._shape_dtype(
                        f"{MODEL_PREFIX}layers.{layer}.{relative}_scale"
                    )
                    if dtype not in (torch.float8_e4m3fn, "F8_E4M3") or scale_dtype not in (torch.float16, torch.float32, "F16", "F32"):
                        raise TypeError(f"invalid FP8 storage for layer {layer}: {relative}")
                    if len(shape) != 2 or scale_shape != ((shape[0] + 127) // 128, (shape[1] + 127) // 128):
                        raise ValueError(f"invalid FP8 scale shape for layer {layer}: {relative}")
                    fp8 += 1
                elif layer in (0, 39) and EXPERT_RE.match(relative) and relative.endswith(".weight"):
                    shape, dtype = self._shape_dtype(f"{MODEL_PREFIX}layers.{layer}.{relative}")
                    if len(shape) != 2 or dtype not in (torch.float16, "F16"):
                        raise TypeError(f"layer {layer} FP16 expert is malformed: {relative}")
                    fp16_experts += 1
        if fp8 != 130:
            raise ValueError(f"expected 130 FP8 matrices, found {fp8}")
        if nvfp4 != 29184:
            raise ValueError(f"expected 29184 NVFP4 matrices, found {nvfp4}")
        if fp16_experts != 1536:
            raise ValueError(f"expected 1536 FP16 routed expert matrices, found {fp16_experts}")
        return CheckpointAudit(fp8, nvfp4, fp16_experts, len(self.layer_ids))

    def _read_layer(self, layer: int, device: Optional[torch.device | str]) -> Dict[str, torch.Tensor]:
        if layer not in self._by_layer:
            raise KeyError(f"unknown layer {layer}")
        by_shard: Dict[str, list[str]] = {}
        for relative, shard in self._by_layer[layer].items():
            by_shard.setdefault(shard, []).append(relative)
        result: Dict[str, torch.Tensor] = {}
        for shard, relatives in by_shard.items():
            with safe_open(str(self.model_dir / shard), framework="pt", device="cpu") as reader:
                for relative in relatives:
                    tensor = reader.get_tensor(f"{MODEL_PREFIX}layers.{layer}.{relative}")
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
    def _stack_experts(raw: Mapping[str, torch.Tensor], projected: str, kind: str) -> Weight:
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
        n, packed_k = data.shape[-2:]
        return Weight(
            "nvfp4", data, (256, n, packed_k * 2), local,
            torch.stack(globals_), torch.stack(inputs),
        )

    @staticmethod
    def _pack_experts(raw: Dict[str, torch.Tensor], layer: int) -> Dict[str, Weight]:
        kind = "fp16" if layer in (0, 39) else "nvfp4"
        gate = Qwen35Checkpoint._stack_experts(raw, "gate_proj", kind)
        up = Qwen35Checkpoint._stack_experts(raw, "up_proj", kind)
        down = Qwen35Checkpoint._stack_experts(raw, "down_proj", kind)
        if kind == "fp16":
            gate_up = Weight("fp16", torch.cat((gate.data, up.data), dim=1), (256, gate.n + up.n, gate.k))
        else:
            if not torch.equal(gate.input_global_scale, up.input_global_scale):
                raise ValueError(f"layer {layer}: gate/up input global scales differ")
            if not torch.equal(gate.global_scale, up.global_scale):
                raise ValueError(f"layer {layer}: gate/up weight global scales differ")
            if not torch.equal(
                gate.input_global_scale,
                gate.input_global_scale[:1].expand_as(gate.input_global_scale),
            ):
                raise ValueError(f"layer {layer}: gate/up input global scale must be static across experts")
            gate_up = Weight(
                "nvfp4", torch.cat((gate.data, up.data), dim=1), (256, gate.n + up.n, gate.k),
                torch.cat((gate.block_scale, up.block_scale), dim=1), gate.global_scale, gate.input_global_scale,
            )
        return {"mlp.experts.gate_up_proj": gate_up, "mlp.experts.down_proj": down}

    @staticmethod
    def _pack_shared(raw: Dict[str, torch.Tensor]) -> Dict[str, Weight]:
        gate = raw.pop("mlp.shared_expert.gate_proj.weight")
        up = raw.pop("mlp.shared_expert.up_proj.weight")
        down = raw.pop("mlp.shared_expert.down_proj.weight")
        return {
            "mlp.shared_expert.gate_up_proj": Weight("fp16", torch.cat((gate, up), dim=0), (gate.shape[0] + up.shape[0], gate.shape[1])),
            "mlp.shared_expert.down_proj": Weight("fp16", down, tuple(down.shape)),
        }

    @staticmethod
    def _move_weight(weight: Weight, device: torch.device | str) -> Weight:
        return Weight(
            weight.kind, weight.data.to(device, non_blocking=True), weight.logical_shape,
            None if weight.block_scale is None else weight.block_scale.to(device, non_blocking=True),
            None if weight.global_scale is None else weight.global_scale.to(device, non_blocking=True),
            None if weight.input_global_scale is None else weight.input_global_scale.to(device, non_blocking=True),
        )

    def load_layer(self, layer: int, device: Optional[torch.device | str] = None) -> Dict[str, torch.Tensor | Weight]:
        """Load one layer, pack on CPU, then make one compact device copy.

        Packing compressed experts directly on GPU creates raw plus stacked plus
        concatenated copies at once.  CPU packing keeps the V100 resident set
        to the final compact layer representation.
        """
        raw = self._read_layer(layer, None)
        output: Dict[str, torch.Tensor | Weight] = self._pack_experts(raw, layer)
        output.update(self._pack_shared(raw))
        expert_prefix = "mlp.experts."
        for key in list(raw):
            if key.startswith(expert_prefix):
                raw.pop(key)
        for key, value in raw.items():
            if key.endswith(".weight") and key + "_scale" in raw:
                base = key[: -len(".weight")]
                output[base] = Weight("fp8", self._raw_fp8(value), tuple(value.shape), raw[key + "_scale"])
            elif key.endswith(".weight_scale"):
                continue
            elif key.endswith(".weight") and value.ndim == 2:
                output[key[: -len(".weight")]] = Weight("fp16", value, tuple(value.shape))
            else:
                output[key] = value
        if device is None:
            return output
        return {
            key: self._move_weight(value, device) if isinstance(value, Weight)
            else value.to(device, non_blocking=True)
            for key, value in output.items()
        }
