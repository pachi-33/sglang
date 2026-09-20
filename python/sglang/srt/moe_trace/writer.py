"""Asynchronous, crash-safe writer for decode-time MoE traces.

The writer deliberately accepts the scheduler's already-host-resident
``MoeTraceBatchOutput``.  It never performs a device-to-host copy and never
serializes Python objects: every chunk is a safetensors file plus JSON
metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from sglang.srt.moe_trace.types import MoeTraceBatchOutput, MoeTraceSiteLayout

FORMAT_ID = "sglang.moe_trace"
FORMAT_VERSION = 1
_STATUSES = {"open", "complete", "failed", "truncated"}


def safe_request_id(request_id: str) -> str:
    """Return a readable, deterministic directory name for an opaque request id.

    The digest is always included, so sanitisation cannot turn two distinct
    request ids into the same directory and path components can never escape
    the trace root.
    """
    if not isinstance(request_id, str):
        raise TypeError("request_id must be a string")
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", request_id).strip(".-_")
    # Do not leave traversal-looking components in the readable part either.
    # (They are harmless after joining, but make directory inspection safer.)
    while ".." in prefix:
        prefix = prefix.replace("..", "-")
    prefix = prefix[:64] or "request"
    return f"{prefix}-{digest}"


def _require_safetensors():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # keep importing the tracing feature optional
        raise RuntimeError(
            "MoE trace writing requires the optional 'safetensors' package"
        ) from exc
    return save_file


class MoeTraceWriter:
    """Write one safetensors chunk per request from a scheduler batch.

    ``submit`` returns ``True`` if accepted and ``False`` when a full queue is
    configured with ``overflow='drop'``.  In both cases ownership of the
    payload is transferred and its ``release()`` method is called exactly once.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        layouts: Sequence[MoeTraceSiteLayout],
        *,
        rank: int = 0,
        queue_depth: int = 8,
        overflow: str = "block",
        fsync: bool = True,
        quantization: Optional[dict[str, Any]] = None,
        format_id: str = FORMAT_ID,
        format_version: int = FORMAT_VERSION,
    ) -> None:
        if queue_depth < 1:
            raise ValueError("queue_depth must be at least one")
        if overflow not in {"block", "drop"}:
            raise ValueError("overflow must be 'block' or 'drop'")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.output_dir = Path(output_dir).resolve()
        self.layouts = tuple(layouts)
        self.rank = rank
        self.overflow = overflow
        self.fsync = fsync
        self.quantization = dict(quantization or {})
        self.format_id = format_id
        self.format_version = format_version
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_depth)
        self._stop = object()
        self._lock = threading.Lock()
        self._worker_error: Optional[BaseException] = None
        self._closed = False
        self._sequences: dict[str, int] = defaultdict(int)
        self._open_requests: set[str] = set()
        self._finalized: set[str] = set()
        self._stats = {
            "submitted_rows": 0,
            "written_rows": 0,
            "written_chunks": 0,
            "dropped_rows": 0,
            "dropped_chunks": 0,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="moe-trace-writer", daemon=True
        )
        self._thread.start()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("MoE trace writer worker failed") from error

    @staticmethod
    def _row_count(payload: MoeTraceBatchOutput) -> int:
        rows = len(payload.request_ids)
        if (
            payload.input_token_ids.ndim == 0
            or payload.input_token_ids.shape[0] != rows
        ):
            raise ValueError("input_token_ids must have one row for each request_id")
        return rows

    def submit(self, payload: MoeTraceBatchOutput) -> bool:
        """Queue a D2H-complete payload, retaining its ownership until consumed."""
        try:
            self._raise_if_failed()
            if self._closed:
                raise RuntimeError("MoE trace writer is closed")
            rows = self._row_count(payload)
            with self._lock:
                finalized = set(self._finalized)
            if any(request_id in finalized for request_id in payload.request_ids):
                raise RuntimeError("cannot submit rows for a finalized request")
            if self.overflow == "drop":
                try:
                    self._queue.put_nowait(payload)
                except queue.Full:
                    with self._lock:
                        self._stats["dropped_rows"] += rows
                        self._stats["dropped_chunks"] += len(set(payload.request_ids))
                    payload.release()
                    return False
            else:
                self._queue.put(payload)
            with self._lock:
                self._stats["submitted_rows"] += rows
            return True
        except BaseException:
            # A rejected payload was never handed to the worker, so release it here.
            # ``release`` itself is idempotent in MoeTraceBatchOutput.
            payload.release()
            raise

    def finalize_request(
        self, request_id: str, *, status: str = "complete", error: str | None = None
    ) -> None:
        """Atomically mark a request's manifest after all earlier queued rows."""
        if status not in _STATUSES or status == "open":
            raise ValueError("final status must be complete, failed, or truncated")
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("MoE trace writer is closed")
        with self._lock:
            if request_id in self._finalized:
                raise RuntimeError("request is already finalized")
            self._finalized.add(request_id)
        task = ("finalize", request_id, status, error)
        try:
            self._queue.put(task)  # finalization must never be dropped
        except BaseException:
            with self._lock:
                self._finalized.discard(request_id)
            raise

    def close(self) -> None:
        """Drain pending work and surface any background exception."""
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._queue.put(self._stop)
        self._thread.join()
        self._raise_if_failed()

    def __enter__(self) -> "MoeTraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is self._stop:
                    return
                with self._lock:
                    failed = self._worker_error is not None
                if isinstance(task, MoeTraceBatchOutput):
                    try:
                        if not failed:
                            self._write_payload(task)
                    except BaseException as exc:
                        with self._lock:
                            if self._worker_error is None:
                                self._worker_error = exc
                        # The trace may be inspected long after the scheduler
                        # has gone away, so leave an on-disk failure marker too.
                        self._persist_payload_failure(task, exc)
                    finally:
                        task.release()
                else:
                    _, request_id, status, error = task  # type: ignore[misc]
                    if not failed:
                        try:
                            self._write_manifest(request_id, status=status, error=error)
                        except BaseException as exc:
                            with self._lock:
                                if self._worker_error is None:
                                    self._worker_error = exc
            finally:
                self._queue.task_done()

    def _persist_payload_failure(
        self, payload: MoeTraceBatchOutput, exc: BaseException
    ) -> None:
        for request_id in set(payload.request_ids):
            try:
                self._write_manifest(
                    request_id, status="failed", error=f"{type(exc).__name__}: {exc}"
                )
            except BaseException:
                # The original error is the one callers need; a broken disk can
                # make the failure marker impossible to write.
                pass

    def _request_dir(self, request_id: str) -> Path:
        path = self.output_dir / safe_request_id(request_id)
        # Belt-and-braces check protecting future changes to the safe-name function.
        if self.output_dir not in path.resolve().parents:
            raise ValueError("unsafe request id")
        return path

    def _site_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "site_id": layout.site.site_id,
                "module_path": layout.site.module_path,
                "layer_id": layout.site.layer_id,
                "router_type": layout.site.router_type,
                "hidden_size": layout.site.hidden_size,
                "num_experts": layout.site.num_experts,
                "top_k": layout.site.top_k,
                "activation_q_offset": layout.activation_q_offset,
                "activation_q_width": layout.activation_q_width,
                "activation_scale_offset": layout.activation_scale_offset,
                "activation_scale_width": layout.activation_scale_width,
                "route_offset": layout.route_offset,
            }
            for layout in self.layouts
        ]

    @staticmethod
    def _cpu_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(
                f"{name} must be CPU; scheduler D2H must finish before submit"
            )
        return tensor.contiguous()

    def _rows(
        self, tensor: torch.Tensor, indices: torch.Tensor, rows: int, name: str
    ) -> torch.Tensor:
        tensor = self._cpu_contiguous(tensor, name)
        if tensor.ndim == 0 or tensor.shape[0] != rows:
            raise ValueError(f"{name} must have one row for each request_id")
        return tensor.index_select(0, indices).contiguous()

    def _flat_site_slice(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        rows: int,
        offset: int,
        width: int,
        name: str,
    ) -> torch.Tensor:
        values = self._rows(tensor, indices, rows, name)
        if values.ndim < 2:
            raise ValueError(f"{name} must be a row-major flat tensor")
        # Only its final dimension is an on-wire flat buffer.  This also lets
        # a producer use an arbitrary leading per-row shape in the future.
        if offset < 0 or width < 0 or offset + width > values.shape[-1]:
            raise ValueError(f"{name} layout offset/width is outside its flat buffer")
        return values[..., offset : offset + width].contiguous()

    def _build_tensors(
        self, payload: MoeTraceBatchOutput, row_numbers: list[int]
    ) -> dict[str, torch.Tensor]:
        rows = self._row_count(payload)
        indices = torch.tensor(row_numbers, dtype=torch.long, device="cpu")
        tensors: dict[str, torch.Tensor] = {
            "input_token_ids": self._rows(
                payload.input_token_ids, indices, rows, "input_token_ids"
            ),
            "positions": self._rows(payload.positions, indices, rows, "positions"),
            "site_valid": self._rows(payload.site_valid, indices, rows, "site_valid"),
        }
        for layout in self.layouts:
            site = layout.site.site_id
            if payload.activation_q is not None:
                tensors[f"site.{site}.activation_q"] = self._flat_site_slice(
                    payload.activation_q,
                    indices,
                    rows,
                    layout.activation_q_offset,
                    layout.activation_q_width,
                    "activation_q",
                )
            if payload.activation_scales is not None:
                tensors[f"site.{site}.scales"] = self._flat_site_slice(
                    payload.activation_scales,
                    indices,
                    rows,
                    layout.activation_scale_offset,
                    layout.activation_scale_width,
                    "activation_scales",
                )
            if payload.expert_ids is not None:
                tensors[f"site.{site}.expert_ids"] = self._flat_site_slice(
                    payload.expert_ids,
                    indices,
                    rows,
                    layout.route_offset,
                    layout.site.top_k,
                    "expert_ids",
                )
            if payload.expert_weights is not None:
                tensors[f"site.{site}.expert_weights"] = self._flat_site_slice(
                    payload.expert_weights,
                    indices,
                    rows,
                    layout.route_offset,
                    layout.site.top_k,
                    "expert_weights",
                )
        return tensors

    def _write_payload(self, payload: MoeTraceBatchOutput) -> None:
        rows = self._row_count(payload)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, request_id in enumerate(payload.request_ids):
            grouped[request_id].append(index)
        for request_id, row_numbers in grouped.items():
            tensors = self._build_tensors(payload, row_numbers)
            self._write_chunk(request_id, tensors)
            with self._lock:
                self._stats["written_rows"] += len(row_numbers)
                self._stats["written_chunks"] += 1

    def _write_chunk(self, request_id: str, tensors: dict[str, torch.Tensor]) -> None:
        request_dir = self._request_dir(request_id)
        rank_dir = request_dir / f"rank-{self.rank:03d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        sequence = self._sequences[request_id]
        self._sequences[request_id] += 1
        filename = f"chunk-{sequence:06d}.safetensors"
        path = rank_dir / filename
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            _require_safetensors()(tensors, str(partial))
            self._sync_file(partial)
            os.replace(partial, path)
            self._sync_dir(rank_dir)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        record = {
            "path": str(Path(rank_dir.name) / filename),
            "rows": int(next(iter(tensors.values())).shape[0]),
            "size_bytes": path.stat().st_size,
            "sha256": self._sha256(path),
            "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        }
        self._write_manifest(request_id, status="open", add_chunk=record)
        self._open_requests.add(request_id)

    def _manifest_path(self, request_id: str) -> Path:
        return self._request_dir(request_id) / "manifest.json"

    def _read_manifest(self, request_id: str) -> dict[str, Any]:
        path = self._manifest_path(request_id)
        if not path.exists():
            return {
                "format": {"id": self.format_id, "version": self.format_version},
                "request_id": request_id,
                "rank": self.rank,
                "status": "open",
                "sites": self._site_metadata(),
                "quantization": self.quantization,
                "feature_flags": {},
                "chunks": [],
                "total_rows": 0,
                "error": None,
            }
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_manifest(
        self,
        request_id: str,
        *,
        status: str,
        error: str | None = None,
        add_chunk: dict[str, Any] | None = None,
    ) -> None:
        request_dir = self._request_dir(request_id)
        request_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest(request_id)
        if add_chunk is not None:
            manifest["chunks"].append(add_chunk)
            manifest["total_rows"] = sum(
                int(chunk["rows"]) for chunk in manifest["chunks"]
            )
            keys = set(add_chunk["shapes"])
            flags = manifest.setdefault("feature_flags", {})
            flags.update(
                {
                    "activations": any(key.endswith(".activation_q") for key in keys),
                    "routes": any(key.endswith(".expert_ids") for key in keys),
                    "scales": any(key.endswith(".scales") for key in keys),
                }
            )
        manifest["status"] = status
        if error is not None:
            manifest["error"] = str(error)
        path = self._manifest_path(request_id)
        partial = path.with_suffix(".json.partial")
        try:
            with partial.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            os.replace(partial, path)
            self._sync_dir(request_dir)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def _sync_file(self, path: Path) -> None:
        if not self.fsync:
            return
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    def _sync_dir(self, path: Path) -> None:
        if not self.fsync:
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
