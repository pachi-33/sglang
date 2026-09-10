"""Opt-in CUDA event timing for one-token Qwen3.5 decode steps."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

DECODE_TIMING_FORMAT = "SGLANG-QWEN35-DECODE-TIMING-v1"
DECODE_TIMING_BOUNDARIES = (
    "layer_start",
    "router_start",
    "router_ready",
    "routed_expert_start",
    "layer_end",
)


class DecodeTimingCapture:
    """Reuse CUDA events while timing consecutive decode steps.

    The caller records every boundary on one compute stream and synchronizes
    once after the complete layer slice.  ``finish_step`` then converts the
    events to compact millisecond offsets without introducing layer-local
    synchronization.
    """

    def __init__(
        self,
        layer_ids: Iterable[int],
        *,
        device: str | torch.device,
    ) -> None:
        ids = tuple(layer_ids)
        if not ids or tuple(sorted(set(ids))) != ids:
            raise ValueError("decode timing layer_ids must be unique and ascending")
        self.layer_ids = ids
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("decode timing requires a CUDA device")
        self._origin = torch.cuda.Event(enable_timing=True)
        self._events = {
            layer_id: {
                boundary: torch.cuda.Event(enable_timing=True)
                for boundary in DECODE_TIMING_BOUNDARIES
            }
            for layer_id in ids
        }
        self._active = False
        self._layer_index = 0
        self._boundary_index = 0
        self.last_result: dict[str, Any] | None = None

    def begin_step(self) -> None:
        if self._active:
            raise RuntimeError("decode timing step is already active")
        self._active = True
        self._layer_index = 0
        self._boundary_index = 0
        self.last_result = None
        self._origin.record()

    def record(self, layer_id: int, boundary: str) -> None:
        if not self._active:
            raise RuntimeError("decode timing step is not active")
        if self._layer_index >= len(self.layer_ids):
            raise RuntimeError("decode timing received too many layer events")
        expected_layer = self.layer_ids[self._layer_index]
        expected_boundary = DECODE_TIMING_BOUNDARIES[self._boundary_index]
        if layer_id != expected_layer or boundary != expected_boundary:
            raise RuntimeError(
                "decode timing event order mismatch: expected "
                f"({expected_layer}, {expected_boundary}), got ({layer_id}, {boundary})"
            )
        self._events[layer_id][boundary].record()
        self._boundary_index += 1
        if self._boundary_index == len(DECODE_TIMING_BOUNDARIES):
            self._boundary_index = 0
            self._layer_index += 1

    def finish_step(self) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("decode timing step is not active")
        if self._layer_index != len(self.layer_ids) or self._boundary_index != 0:
            raise RuntimeError("decode timing step has incomplete layer events")
        elapsed_ms: list[list[float]] = []
        for layer_id in self.layer_ids:
            row = [
                float(self._origin.elapsed_time(self._events[layer_id][boundary]))
                for boundary in DECODE_TIMING_BOUNDARIES
            ]
            if any(not math.isfinite(value) or value < 0.0 for value in row):
                raise RuntimeError("decode timing produced an invalid CUDA duration")
            if any(left > right for left, right in zip(row, row[1:])):
                raise RuntimeError("decode timing boundaries are not monotonic")
            elapsed_ms.append(row)
        result: dict[str, Any] = {
            "format": DECODE_TIMING_FORMAT,
            "layer_ids": list(self.layer_ids),
            "boundaries": list(DECODE_TIMING_BOUNDARIES),
            "elapsed_ms": elapsed_ms,
        }
        self._active = False
        self.last_result = result
        return result

    def abort_step(self) -> None:
        self._active = False
        self._layer_index = 0
        self._boundary_index = 0
        self.last_result = None


__all__ = [
    "DECODE_TIMING_BOUNDARIES",
    "DECODE_TIMING_FORMAT",
    "DecodeTimingCapture",
]
