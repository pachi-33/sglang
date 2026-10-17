"""Forward-aligned IDs-only trace chunks for concurrent target requests."""

from __future__ import annotations

import json
import os
import queue
import time
from pathlib import Path

import torch

from sglang.srt.moe_trace.types import MoeTraceBatchOutput
from sglang.srt.moe_trace.writer import FORMAT_ID, MoeTraceWriter, _require_safetensors

BATCH_FORMAT_VERSION = 2


class MoeTraceBatchWriter(MoeTraceWriter):
    """One bounded writer queue; chunks contain complete target forwards.

    The caller owns the output directory for one run. Existing manifests are
    rejected instead of overwriting another run. Request completion is metadata:
    it must not reject an already-executed overlap forward arriving afterwards.
    """

    def __init__(
        self,
        output_dir,
        layouts,
        *,
        chunk_bytes=8 << 20,
        chunk_forwards=32,
        flush_interval=0.5,
        max_rows_per_request=0,
        **kwargs,
    ):
        if chunk_bytes <= 0 or chunk_forwards <= 0 or flush_interval <= 0:
            raise ValueError("batch trace chunk limits must be positive")
        if kwargs.get("queue_depth", 8) < 1 or kwargs.get("rank", 0) < 0:
            raise ValueError("invalid batch trace queue depth or rank")
        if kwargs.get("overflow", "block") not in {"block", "drop"}:
            raise ValueError("invalid batch trace overflow policy")
        _require_safetensors()
        self.chunk_bytes = chunk_bytes
        self.chunk_forwards = chunk_forwards
        self.flush_interval = flush_interval
        self.max_rows_per_request = max_rows_per_request
        self._pending = []
        self._pending_bytes = 0
        self._last_flush = time.monotonic()
        self._chunks = []
        self._request_status = {}
        self._last_forward_id = -1
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        # Claim the run before the background worker can start.
        with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "format": {"id": FORMAT_ID, "version": BATCH_FORMAT_VERSION},
                    "status": "open",
                    "chunks": [],
                },
                handle,
            )
        super().__init__(output_dir, layouts, **kwargs)
        self._write_run_manifest("open")

    def submit(self, payload: MoeTraceBatchOutput) -> bool:
        try:
            self._raise_if_failed()
            if self._closed:
                raise RuntimeError("MoE trace writer is closed")
            rows = self._row_count(payload)
            if rows == 0:
                raise ValueError("empty batch trace payload")
            layout = payload.row_layout
            if layout is None or layout.expanded_request_ids() != payload.request_ids:
                raise ValueError("batch trace payload requires matching row ownership")
            if payload.expert_ids is None or any(
                value is not None
                for value in (
                    payload.expert_weights,
                    payload.activation_q,
                    payload.activation_scales,
                )
            ):
                raise ValueError("batch trace payload must contain expert IDs only")
            if self.overflow == "drop":
                try:
                    self._queue.put_nowait(payload)
                except queue.Full:
                    with self._lock:
                        self._stats["dropped_rows"] += rows
                        self._stats["dropped_chunks"] += 1
                    payload.release()
                    return False
            else:
                self._queue.put(payload)
            with self._lock:
                self._stats["submitted_rows"] += rows
            return True
        except BaseException:
            payload.release()
            raise

    def finalize_request(self, request_id, *, status="complete", error=None):
        if status not in {"complete", "failed", "truncated"}:
            raise ValueError("invalid request completion status")
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("MoE trace writer is closed")
        self._queue.put(("finalize", request_id, status, error))

    def _run(self):
        while True:
            try:
                task = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                if self._pending:
                    self._try_flush()
                continue
            try:
                if task is self._stop:
                    self._try_flush()
                    self._try_manifest("failed" if self._worker_error else "complete")
                    return
                if isinstance(task, MoeTraceBatchOutput):
                    if self._worker_error is not None:
                        task.release()
                        continue
                    try:
                        if task.row_layout.forward_id <= self._last_forward_id:
                            raise ValueError(
                                "target trace forwards must arrive in order"
                            )
                        self._last_forward_id = task.row_layout.forward_id
                        self._pending.append(task)
                        self._pending_bytes += sum(
                            value.numel() * value.element_size()
                            for value in (
                                task.input_token_ids,
                                task.positions,
                                task.site_valid,
                                task.expert_ids,
                            )
                        )
                    except BaseException as exc:
                        task.release()
                        self._record_failure(exc)
                else:
                    _, request_id, status, error = task
                    self._request_status[request_id] = {
                        "status": status,
                        "error": error,
                    }
                if self._pending and (
                    len(self._pending) >= self.chunk_forwards
                    or self._pending_bytes >= self.chunk_bytes
                    or time.monotonic() - self._last_flush >= self.flush_interval
                ):
                    self._try_flush()
            finally:
                self._queue.task_done()

    def _record_failure(self, exc):
        with self._lock:
            if self._worker_error is None:
                self._worker_error = exc
        self._try_manifest("failed")

    def _try_manifest(self, status):
        try:
            self._write_run_manifest(status)
        except BaseException as exc:
            with self._lock:
                if self._worker_error is None:
                    self._worker_error = exc

    def _try_flush(self):
        pending, self._pending = self._pending, []
        self._pending_bytes = 0
        self._last_flush = time.monotonic()
        try:
            if pending and self._worker_error is None:
                self._write_frames(pending)
        except BaseException as exc:
            self._record_failure(exc)
        finally:
            for payload in pending:
                payload.release()

    def _write_frames(self, payloads):
        frames, pieces = [], []
        row_start = 0
        for payload in payloads:
            rows = self._row_count(payload)
            tensors = self._build_tensors(payload, list(range(rows)))
            layout = payload.row_layout
            tensors["candidate_indices"] = torch.tensor(
                layout.candidate_indices, dtype=torch.int32
            )
            for index, site in enumerate(self.layouts):
                key = f"site.{site.site.site_id}.expert_ids"
                tensors[key] = tensors[key].masked_fill(
                    ~tensors["site_valid"][:, index, None], -1
                )
            pieces.append(tensors)
            frames.append(
                {
                    "forward_id": layout.forward_id,
                    "phase": layout.phase,
                    "row_start": row_start,
                    "rows": rows,
                    "request_ids": list(layout.request_ids),
                    "row_splits": list(layout.row_splits),
                    "capture_indices": list(layout.capture_indices),
                }
            )
            row_start += rows
        tensors = {
            name: torch.cat([piece[name] for piece in pieces], dim=0)
            for name in pieces[0]
        }
        rank_dir = self.output_dir / f"rank-{self.rank:03d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        path = rank_dir / f"chunk-{len(self._chunks):06d}.safetensors"
        partial = path.with_suffix(".safetensors.partial")
        try:
            _require_safetensors()(tensors, str(partial))
            self._sync_file(partial)
            os.replace(partial, path)
            self._sync_dir(rank_dir)
        finally:
            partial.unlink(missing_ok=True)
        metadata_path = path.with_suffix(".meta.json")
        self._write_json(metadata_path, {"forwards": frames})
        record = {
            "path": str(path.relative_to(self.output_dir)),
            "metadata_path": str(metadata_path.relative_to(self.output_dir)),
            "rows": row_start,
            "size_bytes": path.stat().st_size,
            "sha256": self._sha256(path),
            "metadata_sha256": self._sha256(metadata_path),
            "shapes": {name: list(value.shape) for name, value in tensors.items()},
        }
        self._chunks.append(record)
        with self._lock:
            self._stats["written_rows"] += row_start
            self._stats["written_chunks"] += 1
        self._write_run_manifest("open")

    def _write_json(self, path, value):
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            with partial.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            os.replace(partial, path)
            self._sync_dir(path.parent)
        finally:
            partial.unlink(missing_ok=True)

    def _write_run_manifest(self, status):
        self._write_json(
            self.output_dir / "manifest.json",
            {
                "format": {"id": FORMAT_ID, "version": BATCH_FORMAT_VERSION},
                "layout": "batch",
                "model_role": "target",
                "row_policy": "all_valid_candidates",
                "id_space": "logical",
                "max_rows_per_request": self.max_rows_per_request,
                "rank": self.rank,
                "sites": self._site_metadata(),
                "status": status,
                "chunks": self._chunks,
                "requests": self._request_status,
                "total_rows": sum(chunk["rows"] for chunk in self._chunks),
                "stats": self.stats,
                "error": str(self._worker_error) if self._worker_error else None,
            },
        )
