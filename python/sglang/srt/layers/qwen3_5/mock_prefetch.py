# SPDX-License-Identifier: Apache-2.0
"""Deterministic route-oracle and mock expert prediction infrastructure.

The predictor in this module deliberately knows nothing about CUDA cache slots
or transfers.  It turns a route oracle captured by a first deterministic pass
into a fixed candidate table; :class:`MockPrefetchScheduler` only maps a
source-layer router boundary onto a later target layer.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import torch

NUM_LAYERS = 40
NUM_EXPERTS = 256
TOP_K = 8


class MockPrefetchPairConflictError(ValueError):
    """A record/replay pair conflicts with the backend's single pair slot."""


@dataclass(frozen=True)
class MockPrefetchConfig:
    route_recall: float
    top_k: int
    lead_layers: int
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("mock prefetch top_k must be a Python int")
        if not 0 <= self.top_k <= NUM_EXPERTS:
            raise ValueError("mock prefetch top_k must be in [0,256]")
        if isinstance(self.lead_layers, bool) or not isinstance(self.lead_layers, int):
            raise TypeError("mock prefetch lead_layers must be a Python int")
        if not 1 <= self.lead_layers <= 38:
            raise ValueError("mock prefetch lead_layers must be in [1,38]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("mock prefetch seed must be a Python int")
        recall = float(self.route_recall)
        if not math.isfinite(recall):
            raise ValueError("mock prefetch route_recall must be finite")
        maximum = min(1.0, self.top_k / TOP_K)
        if not 0.0 <= recall <= maximum:
            raise ValueError(
                "mock prefetch route_recall must be in "
                f"[0,{maximum:g}] for top_k={self.top_k}"
            )
        if self.top_k == 0 and recall != 0.0:
            raise ValueError("top_k=0 requires route_recall=0")
        object.__setattr__(self, "route_recall", recall)


@dataclass(frozen=True)
class MockPrefetchRun:
    phase: Literal["record", "replay"]
    pair_id: str
    config: MockPrefetchConfig | None = None

    def __post_init__(self) -> None:
        if self.phase not in ("record", "replay"):
            raise ValueError("mock prefetch phase must be record or replay")
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ValueError("mock prefetch pair_id must be a nonempty string")
        if len(self.pair_id) > 200:
            raise ValueError("mock prefetch pair_id is too long")
        if self.phase == "record" and self.config is not None:
            raise ValueError("record phase must not include prefetch hyperparameters")
        if self.phase == "replay" and not isinstance(self.config, MockPrefetchConfig):
            raise ValueError("replay phase requires MockPrefetchConfig")


@dataclass(frozen=True)
class PrefetchCommand:
    request_epoch: int
    output_row: int
    trigger_layer: int
    target_layer: int
    candidate_expert_ids: tuple[int, ...]


class ExpertPredictionProvider(Protocol):
    def candidates(self, output_row: int, target_layer: int) -> tuple[int, ...]: ...


class RouteCaptureStep:
    """One GPU-only row capture; commit is allowed after all 40 layers."""

    def __init__(self, owner: "RouteOracleRecorder", row: int, *, prefill: bool):
        self._owner = owner
        self.row = row
        self.prefill = prefill
        self._captured: set[int] = set()

    def capture(self, layer_id: int, ids: torch.Tensor) -> None:
        if layer_id in self._captured:
            raise RuntimeError(f"route layer {layer_id} captured twice")
        if not 0 <= layer_id < NUM_LAYERS:
            raise ValueError("route layer must be in [0,40)")
        if ids.ndim != 2 or ids.shape[1] != TOP_K:
            raise ValueError("router IDs must be [T,8]")
        selected = ids[-1] if self.prefill else ids[0]
        self._owner._buffer[self.row, layer_id].copy_(selected)
        self._captured.add(layer_id)

    def commit(self, sampled_token_id: int) -> None:
        if len(self._captured) != NUM_LAYERS:
            missing = sorted(set(range(NUM_LAYERS)) - self._captured)
            raise RuntimeError(f"route row is incomplete; missing layers {missing}")
        self._owner._commit(self.row, sampled_token_id)


@dataclass(frozen=True)
class RecordedRouteOracle:
    expert_ids: torch.Tensor
    sampled_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.expert_ids.device.type != "cpu":
            raise ValueError("recorded route oracle must be on CPU")
        if self.expert_ids.dtype != torch.uint8:
            raise TypeError("recorded route oracle must use uint8")
        if tuple(self.expert_ids.shape) != (
            len(self.sampled_token_ids),
            NUM_LAYERS,
            TOP_K,
        ):
            raise ValueError("recorded route oracle has an invalid shape")

    @property
    def rows(self) -> int:
        return len(self.sampled_token_ids)


class RouteOracleRecorder:
    """Capture all router IDs on CUDA and perform one D2H at record completion."""

    def __init__(self, max_rows: int, device: torch.device):
        if max_rows <= 0:
            raise ValueError("route recorder max_rows must be positive")
        self._buffer = torch.empty(
            (max_rows, NUM_LAYERS, TOP_K), dtype=torch.uint8, device=device
        )
        self._tokens: list[int] = []

    def begin_step(self, *, prefill: bool) -> RouteCaptureStep:
        row = len(self._tokens)
        if row >= self._buffer.shape[0]:
            raise RuntimeError("route recorder capacity exhausted")
        return RouteCaptureStep(self, row, prefill=prefill)

    def _commit(self, row: int, sampled_token_id: int) -> None:
        if row != len(self._tokens):
            raise RuntimeError("route rows must be committed in order")
        if isinstance(sampled_token_id, bool) or not isinstance(sampled_token_id, int):
            raise TypeError("sampled token ID must be a Python int")
        self._tokens.append(sampled_token_id)

    def finish(self) -> RecordedRouteOracle:
        rows = len(self._tokens)
        values = self._buffer[:rows].to(device="cpu", dtype=torch.uint8)
        return RecordedRouteOracle(values, tuple(self._tokens))


def _stable_score(seed: int, pair_id: str, row: int, layer: int, expert: int) -> bytes:
    value = f"{seed}\0{pair_id}\0{row}\0{layer}\0{expert}".encode("utf-8")
    return hashlib.blake2b(value, digest_size=16).digest()


class DeterministicMockPredictionProvider:
    """Precompute candidates with an exact request-wide Top-8 recall budget."""

    def __init__(
        self,
        oracle: RecordedRouteOracle,
        config: MockPrefetchConfig,
        *,
        pair_id: str,
    ) -> None:
        self.oracle = oracle
        self.config = config
        self.pair_id = pair_id
        targets = tuple(range(max(1, config.lead_layers), 39))
        groups = tuple(
            (row, layer) for row in range(1, oracle.rows) for layer in targets
        )
        total_true = int(math.floor(config.route_recall * TOP_K * len(groups) + 0.5))
        self._table: dict[tuple[int, int], tuple[int, ...]] = {}
        for index, (row, layer) in enumerate(groups):
            before = (index * total_true) // len(groups) if groups else 0
            after = ((index + 1) * total_true) // len(groups) if groups else 0
            hit_count = after - before
            actual = tuple(int(x) for x in oracle.expert_ids[row, layer].tolist())
            ranked_actual = sorted(
                actual,
                key=lambda expert: _stable_score(
                    config.seed, pair_id, row, layer, expert
                ),
            )
            chosen = ranked_actual[:hit_count]
            actual_set = set(actual)
            wrong = sorted(
                (expert for expert in range(NUM_EXPERTS) if expert not in actual_set),
                key=lambda expert: _stable_score(
                    config.seed, pair_id, row, layer, expert
                ),
            )[: config.top_k - hit_count]
            candidates = tuple(
                sorted(
                    (*chosen, *wrong),
                    key=lambda expert: _stable_score(
                        config.seed, pair_id, row, layer, expert
                    ),
                )
            )
            self._table[(row, layer)] = candidates
        self.eligible_groups = len(groups)
        self.true_candidates = total_true
        self.achieved_recall = total_true / (TOP_K * len(groups)) if groups else 0.0

    def candidates(self, output_row: int, target_layer: int) -> tuple[int, ...]:
        return self._table.get((output_row, target_layer), ())


class MockPrefetchScheduler:
    """Translate router boundaries into nonblocking store prefetch commands."""

    def __init__(
        self,
        provider: DeterministicMockPredictionProvider,
        store: Any,
        request_epoch: int,
    ) -> None:
        self.provider = provider
        self.store = store
        self.request_epoch = request_epoch
        self.output_row: int | None = None

    def begin_decode_row(self, output_row: int) -> None:
        if output_row <= 0:
            raise ValueError(
                "mock prefetch only supports decode rows greater than zero"
            )
        self.output_row = output_row

    def on_router_ready(self, trigger_layer: int) -> None:
        if self.output_row is None:
            raise RuntimeError("prefetch scheduler has no active decode row")
        target = trigger_layer + self.provider.config.lead_layers
        if not 1 <= target <= 38:
            return
        candidates = self.provider.candidates(self.output_row, target)
        if not candidates:
            return
        trigger_event = torch.cuda.Event(enable_timing=True)
        trigger_event.record(torch.cuda.current_stream())
        self.store.prefetch(
            PrefetchCommand(
                request_epoch=self.request_epoch,
                output_row=self.output_row,
                trigger_layer=trigger_layer,
                target_layer=target,
                candidate_expert_ids=candidates,
            ),
            trigger_event,
        )


__all__ = [
    "DeterministicMockPredictionProvider",
    "ExpertPredictionProvider",
    "MockPrefetchConfig",
    "MockPrefetchPairConflictError",
    "MockPrefetchRun",
    "MockPrefetchScheduler",
    "PrefetchCommand",
    "RecordedRouteOracle",
    "RouteCaptureStep",
    "RouteOracleRecorder",
]
