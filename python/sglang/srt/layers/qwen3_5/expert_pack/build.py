"""Build a byte-preserving Qwen3.5 NVFP4 expert pack.

The builder maps the source safetensors shards once and materializes at most
one expert payload at a time.  It never dequantizes or requantizes weights.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from safetensors import safe_open

from ..checkpoint import MODEL_PREFIX, Qwen35Checkpoint
from .format import (
    COMPONENTS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPERT_LAYERS,
    EXPERTS_PER_LAYER,
    FORMAT_ID,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MANIFEST_FILENAME,
    NUM_MODEL_LAYERS,
    PACK_FILENAME,
    PACK_SIZE,
    PAYLOAD_SIZE,
    RECORD_COUNT,
    RECORD_STRIDE,
    TOP_K,
    ExpertPackManifest,
    RecordInfo,
)


class TensorSource(Protocol):
    def get_tensor(self, name: str) -> torch.Tensor: ...


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(model_dir: Path) -> tuple[Mapping[str, str], dict[str, Any]]:
    """Authenticate the exact supported checkpoint and list its source shards."""
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    config_sha256 = _sha256_file(config_path)
    index_sha256 = _sha256_file(index_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            "config.json is not the supported Qwen-AgentWorld checkpoint: "
            f"{config_sha256}"
        )
    if index_sha256 != EXPECTED_INDEX_SHA256:
        raise ValueError(
            "model.safetensors.index.json is not the supported Qwen-AgentWorld "
            f"checkpoint: {index_sha256}"
        )
    # Reuse the runner's exact shape/config contract before opening large shards.
    Qwen35Checkpoint.validate_config(model_dir)
    with index_path.open("r", encoding="utf-8") as handle:
        raw_index = json.load(handle)
    weight_map = raw_index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError("checkpoint index has no valid weight_map")
    shard_names = sorted(set(weight_map.values()))
    shards = []
    for name in shard_names:
        if Path(name).name != name:
            raise ValueError(f"checkpoint shard must be a basename: {name!r}")
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append({"file": name, "size": path.stat().st_size})
    if not shards:
        raise ValueError("checkpoint index contains no shards")
    source = {
        "model_dir": str(model_dir),
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "shards": shards,
    }
    return dict(weight_map), source


class CheckpointExpertSource:
    """Seven long-lived safetensors mappings with single-tensor access."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve()
        self.weight_map, self.metadata = _checkpoint_metadata(self.model_dir)
        self._stack: ExitStack | None = None
        self._readers: dict[str, Any] = {}
        self._validate_expert_names()

    @staticmethod
    def tensor_name(layer_id: int, expert_id: int, projection: str, field: str) -> str:
        return (
            f"{MODEL_PREFIX}layers.{layer_id}.mlp.experts.{expert_id}."
            f"{projection}.{field}"
        )

    def _validate_expert_names(self) -> None:
        for layer_id in EXPERT_LAYERS:
            for expert_id in range(EXPERTS_PER_LAYER):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    for field in (
                        "weight_packed",
                        "weight_scale",
                        "weight_global_scale",
                        "input_global_scale",
                    ):
                        name = self.tensor_name(layer_id, expert_id, projection, field)
                        if name not in self.weight_map:
                            raise ValueError(
                                f"checkpoint is missing expert tensor {name}"
                            )

    def __enter__(self) -> "CheckpointExpertSource":
        if self._stack is not None:
            raise RuntimeError("checkpoint expert source is already open")
        stack = ExitStack()
        try:
            for shard in sorted(set(self.weight_map.values())):
                self._readers[shard] = stack.enter_context(
                    safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
                )
        except BaseException:
            stack.close()
            self._readers.clear()
            raise
        self._stack = stack
        return self

    def __exit__(self, *exc_info: object) -> None:
        assert self._stack is not None
        self._stack.close()
        self._stack = None
        self._readers.clear()

    def get_tensor(self, name: str) -> torch.Tensor:
        if self._stack is None:
            raise RuntimeError("checkpoint expert source is not open")
        try:
            shard = self.weight_map[name]
            reader = self._readers[shard]
        except KeyError as exc:
            raise KeyError(f"unknown checkpoint tensor {name}") from exc
        return reader.get_tensor(name)


def _tensor_bytes(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> bytes:
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be read on CPU")
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(
            f"{name} has {(tuple(value.shape), value.dtype)}, expected {(shape, dtype)}"
        )
    raw = value.detach().contiguous().view(torch.uint8).numpy()
    return raw.tobytes(order="C")


def _positive_finite_scale(raw: bytes, name: str) -> None:
    if len(raw) != 4:
        raise ValueError(f"{name} must contain one FP32 value")
    import struct

    value = struct.unpack("<f", raw)[0]
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, found {value!r}")


def encode_expert_payload(source: TensorSource, layer_id: int, expert_id: int) -> bytes:
    """Return one byte-exact v1 payload from twelve source tensors."""
    if sys.byteorder != "little":
        raise RuntimeError("expert-pack v1 construction requires a little-endian host")
    if layer_id not in EXPERT_LAYERS:
        raise ValueError(f"layer {layer_id} is not an offloaded NVFP4 layer")
    if expert_id < 0 or expert_id >= EXPERTS_PER_LAYER:
        raise ValueError(f"expert {expert_id} is out of range")

    def name(projection: str, field: str) -> str:
        return CheckpointExpertSource.tensor_name(
            layer_id, expert_id, projection, field
        )

    def read(
        projection: str,
        field: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> bytes:
        tensor_name = name(projection, field)
        return _tensor_bytes(
            source.get_tensor(tensor_name),
            name=tensor_name,
            shape=shape,
            dtype=dtype,
        )

    gate_data = read("gate_proj", "weight_packed", (512, 1024), torch.uint8)
    up_data = read("up_proj", "weight_packed", (512, 1024), torch.uint8)
    gate_scale = read("gate_proj", "weight_scale", (512, 128), torch.float8_e4m3fn)
    up_scale = read("up_proj", "weight_scale", (512, 128), torch.float8_e4m3fn)
    gate_global = read("gate_proj", "weight_global_scale", (1,), torch.float32)
    up_global = read("up_proj", "weight_global_scale", (1,), torch.float32)
    gate_input = read("gate_proj", "input_global_scale", (1,), torch.float32)
    up_input = read("up_proj", "input_global_scale", (1,), torch.float32)
    if gate_global != up_global:
        raise ValueError(
            f"layer {layer_id} expert {expert_id}: gate/up weight global scales differ"
        )
    if gate_input != up_input:
        raise ValueError(
            f"layer {layer_id} expert {expert_id}: gate/up input global scales differ"
        )
    _positive_finite_scale(gate_global, name("gate_proj", "weight_global_scale"))
    _positive_finite_scale(gate_input, name("gate_proj", "input_global_scale"))

    down_data = read("down_proj", "weight_packed", (2048, 256), torch.uint8)
    down_scale = read("down_proj", "weight_scale", (2048, 32), torch.float8_e4m3fn)
    down_global = read("down_proj", "weight_global_scale", (1,), torch.float32)
    down_input = read("down_proj", "input_global_scale", (1,), torch.float32)
    _positive_finite_scale(down_global, name("down_proj", "weight_global_scale"))
    _positive_finite_scale(down_input, name("down_proj", "input_global_scale"))

    values = {
        "gate_up.data": gate_data + up_data,
        "gate_up.block_scale": gate_scale + up_scale,
        "gate_up.global_scale": gate_global,
        "gate_up.input_global_scale": gate_input,
        "down.data": down_data,
        "down.block_scale": down_scale,
        "down.global_scale": down_global,
        "down.input_global_scale": down_input,
    }
    payload = bytearray(PAYLOAD_SIZE)
    for component_name, spec in COMPONENTS.items():
        raw = values[component_name]
        if len(raw) != spec.nbytes:
            raise AssertionError(
                f"component {component_name} encoded {len(raw)} bytes, expected {spec.nbytes}"
            )
        payload[spec.offset : spec.end] = raw
    return bytes(payload)


def _check_static_layer_input_scale(
    reference: bytes | None,
    payload: bytes,
    *,
    layer_id: int,
    expert_id: int,
) -> bytes:
    """Enforce the grouped-MoE activation-scale invariant within one layer."""
    spec = COMPONENTS["gate_up.input_global_scale"]
    current = payload[spec.offset : spec.end]
    if len(current) != spec.nbytes:
        raise ValueError(
            f"layer {layer_id} expert {expert_id}: truncated input global scale"
        )
    if reference is not None and current != reference:
        raise ValueError(
            f"layer {layer_id}: gate/up input global scale must be static across "
            f"all experts; expert {expert_id} differs"
        )
    return current if reference is None else reference


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _BuildLock:
    """Fail-fast process lock for one artifact output directory.

    The lock inode is intentionally retained next to (not inside) the artifact
    directory.  Removing a flock file can let a third process lock a new inode
    while another process still owns the old one.
    """

    def __init__(self, output_dir: Path):
        self.path = output_dir.parent / f".{output_dir.name}.build.lock"
        self._handle: Any = None

    def __enter__(self) -> "_BuildLock":
        self._handle = self.path.open("a+", encoding="ascii")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"another expert-pack builder owns {self.path}") from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, *exc_info: object) -> None:
        assert self._handle is not None
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def _manifest_dict(
    *, source: Mapping[str, Any], records: list[RecordInfo], pack_sha256: str
) -> dict[str, Any]:
    return {
        "format": FORMAT_ID,
        "complete": True,
        "pack_file": PACK_FILENAME,
        "pack_size": PACK_SIZE,
        "pack_sha256": pack_sha256,
        "record_count": RECORD_COUNT,
        "record_stride": RECORD_STRIDE,
        "payload_size": PAYLOAD_SIZE,
        "alignment": 4096,
        "model": {
            "num_layers": NUM_MODEL_LAYERS,
            "num_experts": EXPERTS_PER_LAYER,
            "top_k": TOP_K,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "offloaded_layers": list(EXPERT_LAYERS),
            "quantization": "nvfp4",
        },
        "components": {
            name: component.to_dict() for name, component in COMPONENTS.items()
        },
        "source": dict(source),
        "records": [record.to_dict() for record in records],
    }


def build_expert_pack(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
    quiet: bool = False,
) -> ExpertPackManifest:
    """Build and atomically publish ``experts.pack`` and ``manifest.json``."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with _BuildLock(output_dir):
        return _build_expert_pack_locked(
            Path(model_dir).resolve(), output_dir, force=force, quiet=quiet
        )


def _build_expert_pack_locked(
    model_dir: Path,
    output_dir: Path,
    *,
    force: bool,
    quiet: bool,
) -> ExpertPackManifest:
    pack_path = output_dir / PACK_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    unique = f"{os.getpid()}.{uuid.uuid4().hex}"
    pack_partial = output_dir / f"{PACK_FILENAME}.{unique}.partial"
    manifest_partial = output_dir / f"{MANIFEST_FILENAME}.{unique}.partial"
    if not force and (pack_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"expert-pack output already exists in {output_dir}; pass force=True to replace"
        )
    source = CheckpointExpertSource(model_dir)
    records: list[RecordInfo] = []
    pack_digest = hashlib.sha256()
    padding = bytes(RECORD_STRIDE - PAYLOAD_SIZE)
    try:
        with source, pack_partial.open("xb", buffering=0) as output:
            for layer_id in EXPERT_LAYERS:
                layer_input_scale: bytes | None = None
                for expert_id in range(EXPERTS_PER_LAYER):
                    expected_offset = len(records) * RECORD_STRIDE
                    if output.tell() != expected_offset:
                        raise AssertionError("expert-pack writer offset drifted")
                    payload = encode_expert_payload(source, layer_id, expert_id)
                    layer_input_scale = _check_static_layer_input_scale(
                        layer_input_scale,
                        payload,
                        layer_id=layer_id,
                        expert_id=expert_id,
                    )
                    output.write(payload)
                    output.write(padding)
                    pack_digest.update(payload)
                    pack_digest.update(padding)
                    records.append(
                        RecordInfo(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            offset=expected_offset,
                            payload_size=PAYLOAD_SIZE,
                            sha256=hashlib.sha256(payload).hexdigest(),
                        )
                    )
                if not quiet:
                    print(
                        f"built expert-pack layer {layer_id}/{EXPERT_LAYERS[-1]}",
                        file=sys.stderr,
                        flush=True,
                    )
            if output.tell() != PACK_SIZE or len(records) != RECORD_COUNT:
                raise AssertionError(
                    f"expert-pack size/count mismatch: {output.tell()}, {len(records)}"
                )
            output.flush()
            os.fsync(output.fileno())

        manifest_data = _manifest_dict(
            source=source.metadata,
            records=records,
            pack_sha256=pack_digest.hexdigest(),
        )
        manifest = ExpertPackManifest.from_dict(manifest_data)
        os.replace(pack_partial, pack_path)
        _fsync_directory(output_dir)
        with manifest_partial.open("x", encoding="utf-8") as output:
            json.dump(manifest.to_dict(), output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(manifest_partial, manifest_path)
        _fsync_directory(output_dir)
        return manifest
    except BaseException:
        for partial in (pack_partial, manifest_partial):
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="atomically replace an existing artifact"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_expert_pack(
        args.model_dir, args.output_dir, force=args.force, quiet=args.quiet
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / MANIFEST_FILENAME).resolve()),
                "pack": str((args.output_dir / PACK_FILENAME).resolve()),
                "pack_size": manifest.pack_size,
                "pack_sha256": manifest.pack_sha256,
                "records": manifest.record_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CheckpointExpertSource", "build_expert_pack", "encode_expert_payload"]
