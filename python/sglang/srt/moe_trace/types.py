from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class MoeTraceSite:
    """Stable description of one routed-MoE decision point."""

    site_id: int
    module_path: str
    layer_id: int
    router_type: str
    hidden_size: int
    num_experts: int
    top_k: int


@dataclass(frozen=True)
class MoeTraceSiteLayout:
    """Flat-buffer offsets assigned to a :class:`MoeTraceSite`."""

    site: MoeTraceSite
    activation_q_offset: int
    activation_q_width: int
    activation_scale_offset: int
    activation_scale_width: int
    route_offset: int


@dataclass(frozen=True)
class TraceRowLayout:
    """Immutable request/row ownership for one real target forward.

    Capture indices address the router buffer; candidate indices are local to
    each request, so different tree branches at the same position stay distinct.
    All layout metadata is host-owned and stays outside accelerator graphs.
    """

    forward_id: int
    phase: str
    request_ids: tuple[str, ...]
    row_splits: tuple[int, ...]
    capture_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]

    def __post_init__(self):
        rows = len(self.capture_indices)
        if any(
            type(value) is not int
            for value in (
                self.forward_id,
                *self.row_splits,
                *self.capture_indices,
                *self.candidate_indices,
            )
        ) or any(not isinstance(value, str) for value in self.request_ids):
            raise ValueError(
                "trace layout requires integer indices and string request IDs"
            )
        if self.phase not in {"decode", "target_verify"} or self.forward_id < 0:
            raise ValueError("invalid trace forward ID or phase")
        if (
            len(self.row_splits) != len(self.request_ids) + 1
            or not self.row_splits
            or self.row_splits[0] != 0
            or self.row_splits[-1] != rows
            or any(a > b for a, b in zip(self.row_splits, self.row_splits[1:]))
        ):
            raise ValueError("trace row splits must partition the request rows")
        if len(self.candidate_indices) != rows or any(
            index < 0 for index in self.capture_indices + self.candidate_indices
        ):
            raise ValueError("invalid trace capture/candidate indices")
        if len(set(self.capture_indices)) != rows:
            raise ValueError("trace capture indices must be unique")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("duplicate request IDs in one trace forward")

    @classmethod
    def from_counts(cls, forward_id, phase, request_ids, counts, capture_indices=None):
        counts = tuple(counts)
        if len(counts) != len(request_ids) or any(count < 0 for count in counts):
            raise ValueError("one non-negative trace row count is required per request")
        splits = [0]
        candidates = []
        for count in counts:
            splits.append(splits[-1] + count)
            candidates.extend(range(count))
        indices = (
            tuple(range(splits[-1]))
            if capture_indices is None
            else tuple(capture_indices)
        )
        return cls(
            forward_id,
            phase,
            tuple(request_ids),
            tuple(splits),
            indices,
            tuple(candidates),
        )

    def expanded_request_ids(self) -> list[str]:
        return [
            request_id
            for request_id, start, end in zip(
                self.request_ids, self.row_splits, self.row_splits[1:]
            )
            for _ in range(start, end)
        ]

    def select_rows(self, selected: list[int]) -> TraceRowLayout:
        if selected != sorted(set(selected)) or any(
            index < 0 or index >= len(self.capture_indices) for index in selected
        ):
            raise ValueError("selected trace rows must be unique and in order")
        counts = [0] * len(self.request_ids)
        for index in selected:
            counts[bisect_right(self.row_splits, index) - 1] += 1
        splits = [0]
        for count in counts:
            splits.append(splits[-1] + count)
        return TraceRowLayout(
            self.forward_id,
            self.phase,
            self.request_ids,
            tuple(splits),
            tuple(self.capture_indices[index] for index in selected),
            tuple(self.candidate_indices[index] for index in selected),
        )


@dataclass(frozen=True)
class MoeTraceForwardContext:
    layout: TraceRowLayout
    input_token_ids: torch.Tensor
    positions: torch.Tensor


@dataclass
class MoeTraceBatchOutput:
    """Device payload produced by one decode forward.

    ``map_device_tensors`` deliberately mirrors other scheduler result holders:
    the scheduler supplies the one lifetime-safe asynchronous D2H primitive.
    Host metadata never enters CUDA graphs.
    """

    request_ids: list[str]
    input_token_ids: torch.Tensor
    positions: torch.Tensor
    site_valid: torch.Tensor
    activation_q: Optional[torch.Tensor] = None
    activation_scales: Optional[torch.Tensor] = None
    expert_ids: Optional[torch.Tensor] = None
    expert_weights: Optional[torch.Tensor] = None
    release_callback: Optional[Callable[[], None]] = None
    row_layout: Optional[TraceRowLayout] = None

    def map_device_tensors(self, fn) -> None:
        self.input_token_ids = fn(self.input_token_ids)
        self.positions = fn(self.positions)
        self.site_valid = fn(self.site_valid)
        if self.activation_q is not None:
            self.activation_q = fn(self.activation_q)
        if self.activation_scales is not None:
            self.activation_scales = fn(self.activation_scales)
        if self.expert_ids is not None:
            self.expert_ids = fn(self.expert_ids)
        if self.expert_weights is not None:
            self.expert_weights = fn(self.expert_weights)

    def release(self) -> None:
        if self.release_callback is not None:
            callback, self.release_callback = self.release_callback, None
            callback()
