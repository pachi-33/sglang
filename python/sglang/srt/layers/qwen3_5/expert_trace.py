# SPDX-License-Identifier: Apache-2.0
"""Per-request logical expert routing traces for Qwen3.5 generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

FORMAT_ID = "SGLANG-QWEN35-EXPERT-TRACE-v1"
NUM_LAYERS = 40
NUM_EXPERTS = 256
TOP_K = 8
PHASE_PREFILL_LAST = 0
PHASE_DECODE = 1

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ExpertTraceConfig:
    """Server-owned destination for explicitly requested expert traces."""

    trace_dir: str | os.PathLike[str]

    def __post_init__(self) -> None:
        trace_dir = Path(self.trace_dir).expanduser().resolve()
        trace_dir.mkdir(parents=True, exist_ok=True)
        if not trace_dir.is_dir():
            raise NotADirectoryError(trace_dir)
        object.__setattr__(self, "trace_dir", trace_dir)


class ExpertTraceStep:
    """One pending output-attribution row populated by all model layers."""

    def __init__(
        self,
        session: "ExpertTraceSession",
        *,
        row: int,
        phase: int,
        model_input_token_id: int,
        model_input_position: int,
    ) -> None:
        self._session = session
        self.row = row
        self.phase = phase
        self.model_input_token_id = model_input_token_id
        self.model_input_position = model_input_position
        self._seen_layers: set[int] = set()

    def capture(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Copy one layer's logical Top-8 IDs into the request CUDA buffer."""
        if self._session.state != "OPEN":
            raise RuntimeError("expert trace is not open")
        if layer_id not in self._session.layer_ids:
            raise ValueError(f"expert trace received unexpected layer {layer_id}")
        if layer_id in self._seen_layers:
            raise RuntimeError(f"expert trace received layer {layer_id} twice")
        if (
            not isinstance(expert_ids, torch.Tensor)
            or expert_ids.ndim != 2
            or expert_ids.shape[1] != TOP_K
            or expert_ids.shape[0] == 0
            or expert_ids.dtype not in (torch.int32, torch.int64)
            or expert_ids.device != self._session.device
        ):
            raise ValueError(
                "expert trace IDs must be nonempty CUDA int32/int64 [T,8] "
                "on the trace device"
            )
        self._session._expert_ids[self.row, layer_id].copy_(expert_ids[-1])
        self._seen_layers.add(layer_id)

    @property
    def complete(self) -> bool:
        return self._seen_layers == self._session.layer_ids


class ExpertTraceSession:
    """Collect and atomically publish one request's output-attribution trace."""

    def __init__(
        self,
        config: ExpertTraceConfig,
        *,
        request_id: str,
        max_rows: int,
        prompt_tokens: int,
        device: torch.device | str,
        identity: Mapping[str, Any],
    ) -> None:
        if not isinstance(config, ExpertTraceConfig):
            raise TypeError("config must be ExpertTraceConfig")
        if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
            raise ValueError(
                "expert trace request_id must be a safe 1..192 character basename"
            )
        for value, label in ((max_rows, "max_rows"), (prompt_tokens, "prompt_tokens")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"expert trace {label} must be a positive integer")

        self.config = config
        self.request_id = request_id
        self.max_rows = max_rows
        self.prompt_tokens = prompt_tokens
        self.device = torch.device(device)
        self.identity = dict(identity)
        self.layer_ids = frozenset(range(NUM_LAYERS))
        self.started_at = _utc_now()
        self.state = "OPEN"
        self._committed_rows = 0
        self._pending: ExpertTraceStep | None = None
        self._sampled_token_ids: list[int] = []
        self._model_input_token_ids: list[int] = []
        self._model_input_positions: list[int] = []
        self._phases: list[int] = []

        base = config.trace_dir / request_id
        self.metadata_path = base.with_name(base.name + ".trace.json")
        self.data_path = base.with_name(base.name + ".trace.npz")
        self._metadata_partial = Path(str(self.metadata_path) + ".partial")
        self._data_partial = Path(str(self.data_path) + ".partial")
        for path in (
            self.metadata_path,
            self.data_path,
            self._metadata_partial,
            self._data_partial,
        ):
            if path.exists():
                raise FileExistsError(f"expert trace output already exists: {path}")

        self._expert_ids = torch.empty(
            (max_rows, NUM_LAYERS, TOP_K),
            dtype=torch.int32,
            device=self.device,
        )

    @property
    def committed_rows(self) -> int:
        return self._committed_rows

    def begin_step(
        self,
        *,
        phase: int,
        model_input_token_id: int,
        model_input_position: int,
    ) -> ExpertTraceStep:
        if self.state != "OPEN":
            raise RuntimeError("expert trace is not open")
        if self._pending is not None:
            raise RuntimeError("expert trace already has a pending row")
        if self._committed_rows >= self.max_rows:
            raise RuntimeError("expert trace row capacity is exhausted")
        if phase not in (PHASE_PREFILL_LAST, PHASE_DECODE):
            raise ValueError("expert trace phase must be prefill-last or decode")
        for value, label in (
            (model_input_token_id, "model_input_token_id"),
            (model_input_position, "model_input_position"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"expert trace {label} must be a nonnegative integer")
        if self._committed_rows == 0:
            if phase != PHASE_PREFILL_LAST:
                raise RuntimeError("the first expert trace row must come from prefill")
        elif phase != PHASE_DECODE:
            raise RuntimeError("expert trace rows after prefill must come from decode")

        step = ExpertTraceStep(
            self,
            row=self._committed_rows,
            phase=phase,
            model_input_token_id=model_input_token_id,
            model_input_position=model_input_position,
        )
        self._pending = step
        return step

    def commit_step(self, step: ExpertTraceStep, sampled_token_id: int) -> None:
        if self.state != "OPEN" or step is not self._pending:
            raise RuntimeError("expert trace step is not the active pending row")
        if not step.complete:
            missing = sorted(self.layer_ids - step._seen_layers)
            raise RuntimeError(f"expert trace step is missing layers: {missing}")
        if (
            isinstance(sampled_token_id, bool)
            or not isinstance(sampled_token_id, int)
            or sampled_token_id < 0
        ):
            raise ValueError("sampled_token_id must be a nonnegative Python int")

        self._sampled_token_ids.append(sampled_token_id)
        self._model_input_token_ids.append(step.model_input_token_id)
        self._model_input_positions.append(step.model_input_position)
        self._phases.append(step.phase)
        self._committed_rows += 1
        self._pending = None

    def _arrays(self) -> dict[str, np.ndarray]:
        rows = self._committed_rows
        expert_ids = self._expert_ids[:rows].detach().cpu().numpy()
        if expert_ids.size and (
            int(expert_ids.min()) < 0 or int(expert_ids.max()) >= NUM_EXPERTS
        ):
            raise ValueError("expert trace contains an out-of-range global expert ID")
        return {
            "expert_ids": expert_ids.astype(np.uint8, copy=False),
            "sampled_token_ids": np.asarray(self._sampled_token_ids, dtype=np.int32),
            "model_input_token_ids": np.asarray(
                self._model_input_token_ids, dtype=np.int32
            ),
            "model_input_positions": np.asarray(
                self._model_input_positions, dtype=np.int32
            ),
            "phase": np.asarray(self._phases, dtype=np.uint8),
        }

    def finalize(
        self,
        *,
        status: str,
        stopped_on_eos: bool,
        error: BaseException | None,
    ) -> dict[str, Any]:
        if self.state != "OPEN":
            raise RuntimeError("expert trace can only be finalized once")
        if status not in ("ok", "failed"):
            raise ValueError("expert trace status must be 'ok' or 'failed'")
        if status == "ok" and error is not None:
            raise ValueError("a successful expert trace cannot contain an error")
        if status == "ok" and (self._pending is not None or not self._committed_rows):
            raise RuntimeError(
                "a successful expert trace requires at least one complete row"
            )
        self.state = "FINALIZING"
        data_published = False
        metadata_published = False
        try:
            arrays = self._arrays()
            with self._data_partial.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self._data_partial, self.data_path)
            data_published = True

            data_size = self.data_path.stat().st_size
            data_sha256 = _sha256_path(self.data_path)
            metadata: dict[str, Any] = {
                "format": FORMAT_ID,
                "trace_id": self.request_id,
                "status": status,
                "started_at": self.started_at,
                "finished_at": _utc_now(),
                "request": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self._committed_rows,
                    "stopped_on_eos": bool(stopped_on_eos),
                    "completed_rows": self._committed_rows,
                },
                "semantics": {
                    "kind": "sampled-output-attribution",
                    "description": (
                        "row 0 uses the last prompt position; row i>0 uses the "
                        "decode input sampled_token_ids[i-1]"
                    ),
                    "phase_values": {
                        str(PHASE_PREFILL_LAST): "prefill_last_position",
                        str(PHASE_DECODE): "decode",
                    },
                },
                "routing": {
                    "identity": "(layer_id, global_expert_id)",
                    "layer_ids": list(range(NUM_LAYERS)),
                    "top_k": TOP_K,
                    "rank_order": "router_topk",
                    "arrays": {
                        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for name, value in arrays.items()
                    },
                    "expert_ids_axes": ["output_token", "layer_id", "topk_rank"],
                },
                "identity": self.identity,
                "artifact": {
                    "file": self.data_path.name,
                    "size": data_size,
                    "sha256": data_sha256,
                },
                "error": (
                    None
                    if error is None
                    else {"type": type(error).__name__, "message": str(error)}
                ),
            }
            with self._metadata_partial.open("x", encoding="utf-8") as handle:
                json.dump(
                    metadata, handle, ensure_ascii=False, indent=2, sort_keys=True
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self._metadata_partial, self.metadata_path)
            metadata_published = True
            _fsync_directory(self.config.trace_dir)
            self.state = "COMMITTED"
            self._pending = None
            return metadata
        except BaseException:
            self.state = "FAILED"
            for path in (self._metadata_partial, self._data_partial):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            if metadata_published:
                try:
                    self.metadata_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if data_published:
                try:
                    self.data_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise


__all__ = [
    "FORMAT_ID",
    "NUM_LAYERS",
    "PHASE_DECODE",
    "PHASE_PREFILL_LAST",
    "TOP_K",
    "ExpertTraceConfig",
    "ExpertTraceSession",
    "ExpertTraceStep",
]
