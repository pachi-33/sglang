"""Safe reader for :mod:`sglang.srt.moe_trace.writer` output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch

from sglang.srt.moe_trace.batch_writer import BATCH_FORMAT_VERSION
from sglang.srt.moe_trace.types import TraceRowLayout
from sglang.srt.moe_trace.writer import FORMAT_ID, FORMAT_VERSION, safe_request_id


def _require_safetensors():
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "MoE trace loading requires the optional 'safetensors' package"
        ) from exc
    return load_file


class MoeTraceLoader:
    """Validate and load trace chunks without using pickle or ``torch.load``."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()

    def manifest(self, request_id: str) -> dict:
        path = self.output_dir / safe_request_id(request_id) / "manifest.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"no MoE trace manifest for request {request_id!r}"
            ) from None
        if (
            not isinstance(manifest, dict)
            or manifest.get("format", {}).get("id") != FORMAT_ID
        ):
            raise ValueError("invalid MoE trace manifest format")
        if manifest["format"].get("version") != FORMAT_VERSION:
            raise ValueError("unsupported MoE trace format version")
        if manifest.get("request_id") != request_id:
            raise ValueError("manifest request id does not match requested trace")
        if manifest.get("status") not in {"open", "complete", "failed", "truncated"}:
            raise ValueError("invalid MoE trace manifest status")
        if not isinstance(manifest.get("chunks"), list):
            raise ValueError("manifest chunks must be a list")
        return manifest

    def iter_chunks(
        self, request_id: str, *, verify: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield validated tensor mappings in manifest order."""
        manifest = self.manifest(request_id)
        request_dir = (self.output_dir / safe_request_id(request_id)).resolve()
        for chunk in manifest["chunks"]:
            if not isinstance(chunk, dict):
                raise ValueError("invalid chunk record")
            relative = chunk.get("path")
            if not isinstance(relative, str):
                raise ValueError("chunk has no path")
            path = (request_dir / relative).resolve()
            if request_dir not in path.parents or path.suffix != ".safetensors":
                raise ValueError("unsafe chunk path in manifest")
            if not path.is_file():
                raise ValueError(f"trace chunk is missing: {relative}")
            if chunk.get("size_bytes") != path.stat().st_size:
                raise ValueError(f"size mismatch for trace chunk {relative}")
            if verify and self._sha256(path) != chunk.get("sha256"):
                raise ValueError(f"sha256 mismatch for trace chunk {relative}")
            tensors = _require_safetensors()(str(path), device="cpu")
            expected = chunk.get("shapes")
            if not isinstance(expected, dict) or set(tensors) != set(expected):
                raise ValueError(
                    f"tensor keys do not match manifest for trace chunk {relative}"
                )
            for name, shape in expected.items():
                if not isinstance(shape, list) or list(tensors[name].shape) != shape:
                    raise ValueError(
                        f"tensor shape mismatch for {name} in trace chunk {relative}"
                    )
                if tensors[name].device.type != "cpu":
                    raise ValueError("safetensors loader returned a non-CPU tensor")
            rows = chunk.get("rows")
            if not isinstance(rows, int) or any(
                tensor.ndim == 0 or tensor.shape[0] != rows
                for tensor in tensors.values()
            ):
                raise ValueError(f"row count mismatch for trace chunk {relative}")
            yield tensors

    def load_trace(
        self, request_id: str, *, verify: bool = True
    ) -> dict[str, torch.Tensor]:
        """Concatenate every chunk for one request along its row dimension."""
        chunks = list(self.iter_chunks(request_id, verify=verify))
        if not chunks:
            return {}
        keys = set(chunks[0])
        if any(set(chunk) != keys for chunk in chunks[1:]):
            raise ValueError("trace chunks have inconsistent tensor keys")
        return {
            name: torch.cat([chunk[name] for chunk in chunks], dim=0)
            for name in sorted(keys)
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class MoeTraceBatchLoader:
    """Read v2 IDs-only chunks, retaining forward and concurrent-request order."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()

    def manifest(self) -> dict:
        with (self.output_dir / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            not isinstance(manifest, dict)
            or manifest.get("format")
            != {"id": FORMAT_ID, "version": BATCH_FORMAT_VERSION}
            or manifest.get("layout") != "batch"
            or manifest.get("model_role") != "target"
            or manifest.get("row_policy") != "all_valid_candidates"
            or manifest.get("id_space") != "logical"
            or manifest.get("status") not in {"open", "complete", "failed"}
            or not isinstance(manifest.get("chunks"), list)
            or not isinstance(manifest.get("sites"), list)
            or not manifest.get("sites")
        ):
            raise ValueError("invalid batch MoE trace manifest")
        return manifest

    def _path(self, relative, suffix):
        if not isinstance(relative, str):
            raise ValueError("batch trace chunk has no path")
        path = (self.output_dir / relative).resolve()
        if self.output_dir not in path.parents or not path.name.endswith(suffix):
            raise ValueError("unsafe batch trace chunk path")
        return path

    def iter_chunks(self, *, verify: bool = True):
        """Yield (forward metadata, CPU tensors) for each committed chunk."""
        manifest = self.manifest()
        previous_forward_id = -1
        for chunk in manifest["chunks"]:
            if not isinstance(chunk, dict):
                raise ValueError("invalid batch trace chunk record")
            path = self._path(chunk.get("path"), ".safetensors")
            metadata_path = self._path(chunk.get("metadata_path"), ".meta.json")
            if not path.is_file() or not metadata_path.is_file():
                raise ValueError("batch trace chunk or metadata is missing")
            if path.stat().st_size != chunk.get("size_bytes"):
                raise ValueError("size mismatch for batch trace chunk")
            if verify and (
                MoeTraceLoader._sha256(path) != chunk.get("sha256")
                or MoeTraceLoader._sha256(metadata_path) != chunk.get("metadata_sha256")
            ):
                raise ValueError("sha256 mismatch for batch trace chunk")
            tensors = _require_safetensors()(str(path), device="cpu")
            shapes = chunk.get("shapes")
            rows = chunk.get("rows")
            if (
                not isinstance(shapes, dict)
                or set(tensors) != set(shapes)
                or not isinstance(rows, int)
            ):
                raise ValueError("invalid batch trace tensor keys or row count")
            if any(
                list(tensor.shape) != shapes[name]
                or tensor.ndim == 0
                or tensor.shape[0] != rows
                for name, tensor in tensors.items()
            ):
                raise ValueError("shape mismatch for batch trace chunk")
            required = {
                "input_token_ids",
                "positions",
                "site_valid",
                "candidate_indices",
            }
            required.update(
                f"site.{site['site_id']}.expert_ids" for site in manifest["sites"]
            )
            if set(tensors) != required:
                raise ValueError("invalid IDs-only trace tensor keys")
            if any(
                tensors[name].shape != (rows,)
                for name in ("input_token_ids", "positions", "candidate_indices")
            ):
                raise ValueError("invalid batch trace row metadata")
            if tensors["candidate_indices"].dtype != torch.int32:
                raise ValueError("invalid batch trace candidate indices")
            if tensors["site_valid"].dtype != torch.bool or tensors[
                "site_valid"
            ].shape != (rows, len(manifest["sites"])):
                raise ValueError("invalid batch trace site validity mask")
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            frames = metadata.get("forwards")
            if not isinstance(frames, list) or not frames:
                raise ValueError("batch trace has no forward metadata")
            row_start = 0
            for frame in frames:
                count = frame.get("rows")
                if (
                    frame.get("row_start") != row_start
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise ValueError("invalid batch trace forward row range")
                layout = TraceRowLayout(
                    frame["forward_id"],
                    frame["phase"],
                    tuple(frame["request_ids"]),
                    tuple(frame["row_splits"]),
                    tuple(frame["capture_indices"]),
                    tuple(
                        tensors["candidate_indices"][
                            row_start : row_start + count
                        ].tolist()
                    ),
                )
                if (
                    len(layout.capture_indices) != count
                    or layout.forward_id <= previous_forward_id
                ):
                    raise ValueError("invalid batch trace forward ordering")
                previous_forward_id = layout.forward_id
                row_start += count
            if row_start != rows:
                raise ValueError("batch trace forward rows do not cover the chunk")
            for site in manifest["sites"]:
                ids = tensors.get(f"site.{site['site_id']}.expert_ids")
                if (
                    ids is None
                    or ids.dtype != torch.int32
                    or ids.shape != (rows, site["top_k"])
                ):
                    raise ValueError("invalid batch trace expert IDs")
            yield metadata, tensors

    def iter_forwards(self, *, verify: bool = True):
        for metadata, tensors in self.iter_chunks(verify=verify):
            for frame in metadata["forwards"]:
                start, count = frame["row_start"], frame["rows"]
                yield (
                    frame,
                    {
                        name: tensor[start : start + count]
                        for name, tensor in tensors.items()
                    },
                )

    def load_trace(
        self, request_id: str, *, verify: bool = True
    ) -> dict[str, torch.Tensor]:
        pieces = []
        for frame, tensors in self.iter_forwards(verify=verify):
            if request_id not in frame["request_ids"]:
                continue
            slot = frame["request_ids"].index(request_id)
            start, end = frame["row_splits"][slot : slot + 2]
            if start == end:
                continue
            piece = {name: tensor[start:end] for name, tensor in tensors.items()}
            piece["forward_ids"] = torch.full(
                (end - start,), frame["forward_id"], dtype=torch.int64
            )
            pieces.append(piece)
        return (
            {name: torch.cat([piece[name] for piece in pieces]) for name in pieces[0]}
            if pieces
            else {}
        )
