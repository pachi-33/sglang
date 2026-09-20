"""Safe reader for :mod:`sglang.srt.moe_trace.writer` output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch

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
