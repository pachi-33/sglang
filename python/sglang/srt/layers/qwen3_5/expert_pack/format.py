"""On-disk contract for the Qwen3.5 NVFP4 expert pack.

The pack intentionally has no in-band header.  Every expert starts at a 4 KiB
aligned, mathematically-derived offset, while the atomically-published JSON
manifest authenticates both the whole file and each useful payload.  Keeping
the data file headerless lets the runtime issue one bounded ``preadv`` per
expert directly into pinned staging memory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAT_ID = "SGLANG-QWEN35-NVFP4-EXPERTPACK-v1"
PACK_FILENAME = "experts.pack"
MANIFEST_FILENAME = "manifest.json"

ALIGNMENT = 4096
FIRST_EXPERT_LAYER = 1
LAST_EXPERT_LAYER = 38
EXPERT_LAYERS = tuple(range(FIRST_EXPERT_LAYER, LAST_EXPERT_LAYER + 1))
NUM_MODEL_LAYERS = 40
EXPERTS_PER_LAYER = 256
TOP_K = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512

PAYLOAD_SIZE = 1_769_488
RECORD_STRIDE = 1_773_568
RECORD_COUNT = len(EXPERT_LAYERS) * EXPERTS_PER_LAYER
PACK_SIZE = RECORD_COUNT * RECORD_STRIDE

EXPECTED_CONFIG_SHA256 = (
    "e1be0a1fb619901c1f97afeb75beb2a6581be706e4eb6166d10690104e0b7634"
)
EXPECTED_INDEX_SHA256 = (
    "5bd3e4596cf3d20483079f01df50c78946a2fe2aa27251acaa13dff3a06e27ed"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ComponentSpec:
    """One typed, contiguous view within an expert payload."""

    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


# Insertion order is the physical payload order used by the builder.
COMPONENTS: dict[str, ComponentSpec] = {
    "gate_up.data": ComponentSpec(0, 1_048_576, (1024, 1024), "uint8"),
    "gate_up.block_scale": ComponentSpec(1_048_576, 131_072, (1024, 128), "uint8"),
    "gate_up.global_scale": ComponentSpec(1_179_648, 4, (1,), "float32-le"),
    "gate_up.input_global_scale": ComponentSpec(1_179_652, 4, (1,), "float32-le"),
    "down.data": ComponentSpec(1_179_656, 524_288, (2048, 256), "uint8"),
    "down.block_scale": ComponentSpec(1_703_944, 65_536, (2048, 32), "uint8"),
    "down.global_scale": ComponentSpec(1_769_480, 4, (1,), "float32-le"),
    "down.input_global_scale": ComponentSpec(1_769_484, 4, (1,), "float32-le"),
}


@dataclass(frozen=True)
class RecordInfo:
    layer_id: int
    expert_id: int
    offset: int
    payload_size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "offset": self.offset,
            "payload_size": self.payload_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ExpertPackManifest:
    """Validated immutable view of ``manifest.json``."""

    format: str
    complete: bool
    pack_file: str
    pack_size: int
    pack_sha256: str
    record_count: int
    record_stride: int
    payload_size: int
    alignment: int
    model: Mapping[str, Any]
    components: Mapping[str, ComponentSpec]
    source: Mapping[str, Any]
    records: tuple[RecordInfo, ...]

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, verify_identity: bool = True
    ) -> "ExpertPackManifest":
        if not isinstance(value, Mapping):
            raise TypeError("expert-pack manifest must be a JSON object")

        expected_scalars = {
            "format": FORMAT_ID,
            "complete": True,
            "pack_file": PACK_FILENAME,
            "pack_size": PACK_SIZE,
            "record_count": RECORD_COUNT,
            "record_stride": RECORD_STRIDE,
            "payload_size": PAYLOAD_SIZE,
            "alignment": ALIGNMENT,
        }
        for key, expected in expected_scalars.items():
            if value.get(key) != expected:
                raise ValueError(
                    f"invalid expert-pack {key}: {value.get(key)!r}, expected {expected!r}"
                )
        pack_sha256 = value.get("pack_sha256")
        if not isinstance(pack_sha256, str) or not _SHA256_RE.fullmatch(pack_sha256):
            raise ValueError("expert-pack pack_sha256 must be lowercase SHA-256")

        model = value.get("model")
        expected_model = {
            "num_layers": NUM_MODEL_LAYERS,
            "num_experts": EXPERTS_PER_LAYER,
            "top_k": TOP_K,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "offloaded_layers": list(EXPERT_LAYERS),
            "quantization": "nvfp4",
        }
        if not isinstance(model, Mapping):
            raise ValueError("expert-pack model must be an object")
        for key, expected in expected_model.items():
            if model.get(key) != expected:
                raise ValueError(f"invalid expert-pack model.{key}: {model.get(key)!r}")

        components_value = value.get("components")
        if not isinstance(components_value, Mapping):
            raise ValueError("expert-pack components must be an object")
        if set(components_value) != set(COMPONENTS):
            raise ValueError("expert-pack component names do not match the v1 ABI")
        parsed_components: dict[str, ComponentSpec] = {}
        for name, expected in COMPONENTS.items():
            raw = components_value[name]
            if not isinstance(raw, Mapping):
                raise ValueError(f"expert-pack component {name} must be an object")
            try:
                parsed = ComponentSpec(
                    offset=int(raw["offset"]),
                    nbytes=int(raw["nbytes"]),
                    shape=tuple(int(item) for item in raw["shape"]),
                    dtype=str(raw["dtype"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed expert-pack component {name}") from exc
            if parsed != expected:
                raise ValueError(
                    f"expert-pack component {name} differs from the v1 ABI"
                )
            parsed_components[name] = parsed

        source = value.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("expert-pack source must be an object")
        for field in ("config_sha256", "index_sha256"):
            digest = source.get(field)
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(
                    f"expert-pack source.{field} must be lowercase SHA-256"
                )
        if verify_identity:
            if source.get("config_sha256") != EXPECTED_CONFIG_SHA256:
                raise ValueError(
                    "expert-pack config identity does not match Qwen-AgentWorld"
                )
            if source.get("index_sha256") != EXPECTED_INDEX_SHA256:
                raise ValueError(
                    "expert-pack index identity does not match Qwen-AgentWorld"
                )
        shards = source.get("shards")
        if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
            raise ValueError("expert-pack source.shards must be an array")
        shard_names: set[str] = set()
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise ValueError("expert-pack source shard must be an object")
            name, size = shard.get("file"), shard.get("size")
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name in shard_names
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise ValueError("invalid or duplicate expert-pack source shard")
            shard_names.add(name)
        if not shards:
            raise ValueError("expert-pack source shard list is empty")

        records_value = value.get("records")
        if not isinstance(records_value, Sequence) or isinstance(
            records_value, (str, bytes)
        ):
            raise ValueError("expert-pack records must be an array")
        if len(records_value) != RECORD_COUNT:
            raise ValueError(
                f"expert-pack has {len(records_value)} records, expected {RECORD_COUNT}"
            )
        parsed_records: list[RecordInfo] = []
        for index, raw in enumerate(records_value):
            if not isinstance(raw, Mapping):
                raise ValueError(f"expert-pack record {index} must be an object")
            layer_id, expert_id = record_key(index)
            try:
                record = RecordInfo(
                    layer_id=int(raw["layer_id"]),
                    expert_id=int(raw["expert_id"]),
                    offset=int(raw["offset"]),
                    payload_size=int(raw["payload_size"]),
                    sha256=str(raw["sha256"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed expert-pack record {index}") from exc
            if (
                record.layer_id != layer_id
                or record.expert_id != expert_id
                or record.offset != index * RECORD_STRIDE
                or record.payload_size != PAYLOAD_SIZE
                or not _SHA256_RE.fullmatch(record.sha256)
            ):
                raise ValueError(f"invalid expert-pack record {index}: {record}")
            parsed_records.append(record)

        return cls(
            format=FORMAT_ID,
            complete=True,
            pack_file=PACK_FILENAME,
            pack_size=PACK_SIZE,
            pack_sha256=pack_sha256,
            record_count=RECORD_COUNT,
            record_stride=RECORD_STRIDE,
            payload_size=PAYLOAD_SIZE,
            alignment=ALIGNMENT,
            model=dict(model),
            components=parsed_components,
            source=dict(source),
            records=tuple(parsed_records),
        )

    def record(self, layer_id: int, expert_id: int) -> RecordInfo:
        return self.records[record_index(layer_id, expert_id)]

    def pack_path(self, manifest_path: str | Path) -> Path:
        return Path(manifest_path).resolve().parent / self.pack_file

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "complete": self.complete,
            "pack_file": self.pack_file,
            "pack_size": self.pack_size,
            "pack_sha256": self.pack_sha256,
            "record_count": self.record_count,
            "record_stride": self.record_stride,
            "payload_size": self.payload_size,
            "alignment": self.alignment,
            "model": dict(self.model),
            "components": {
                name: component.to_dict() for name, component in self.components.items()
            },
            "source": dict(self.source),
            "records": [record.to_dict() for record in self.records],
        }


def record_index(layer_id: int, expert_id: int) -> int:
    if layer_id not in EXPERT_LAYERS:
        raise ValueError(
            f"layer {layer_id} is not offloaded; expected {FIRST_EXPERT_LAYER}..{LAST_EXPERT_LAYER}"
        )
    if expert_id < 0 or expert_id >= EXPERTS_PER_LAYER:
        raise ValueError(f"expert {expert_id} is outside 0..{EXPERTS_PER_LAYER - 1}")
    return (layer_id - FIRST_EXPERT_LAYER) * EXPERTS_PER_LAYER + expert_id


def record_key(index: int) -> tuple[int, int]:
    if index < 0 or index >= RECORD_COUNT:
        raise ValueError(f"record index {index} is outside 0..{RECORD_COUNT - 1}")
    layer_delta, expert_id = divmod(index, EXPERTS_PER_LAYER)
    return FIRST_EXPERT_LAYER + layer_delta, expert_id


def record_offset(layer_id: int, expert_id: int) -> int:
    return record_index(layer_id, expert_id) * RECORD_STRIDE


def load_manifest(
    path: str | Path, *, verify_identity: bool = True
) -> ExpertPackManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid expert-pack JSON: {manifest_path}") from exc
    return ExpertPackManifest.from_dict(raw, verify_identity=verify_identity)


if COMPONENTS[next(reversed(COMPONENTS))].end != PAYLOAD_SIZE:
    raise AssertionError("expert-pack component layout does not fill PAYLOAD_SIZE")
if RECORD_STRIDE % ALIGNMENT or RECORD_STRIDE < PAYLOAD_SIZE:
    raise AssertionError("expert-pack record stride violates alignment")
if PACK_SIZE != 17_253_269_504:
    raise AssertionError("expert-pack fixed size contract changed")


__all__ = [
    "ALIGNMENT",
    "COMPONENTS",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_INDEX_SHA256",
    "EXPERTS_PER_LAYER",
    "EXPERT_LAYERS",
    "FORMAT_ID",
    "MANIFEST_FILENAME",
    "PACK_FILENAME",
    "PACK_SIZE",
    "PAYLOAD_SIZE",
    "RECORD_COUNT",
    "RECORD_STRIDE",
    "ComponentSpec",
    "ExpertPackManifest",
    "RecordInfo",
    "load_manifest",
    "record_index",
    "record_key",
    "record_offset",
]
