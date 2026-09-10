# SPDX-License-Identifier: Apache-2.0
"""Pinned-host expert repository and layer-local GPU LRU cache."""

from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from ..mock_prefetch import MockPrefetchConfig, PrefetchCommand
from ..weights import Weight
from .format import PAYLOAD_SIZE, load_manifest

logger = logging.getLogger(__name__)
_NUM_EXPERTS = 256
_ACTIVE_LAYERS = tuple(range(1, 39))
_COMPONENT_LAYOUT = (
    ("gate_up.data", 0, 1_048_576, (1024, 1024), torch.uint8),
    ("gate_up.block_scale", 1_048_576, 131_072, (1024, 128), torch.uint8),
    ("gate_up.global_scale", 1_179_648, 4, (), torch.float32),
    ("gate_up.input_global_scale", 1_179_652, 4, (), torch.float32),
    ("down.data", 1_179_656, 524_288, (2048, 256), torch.uint8),
    ("down.block_scale", 1_703_944, 65_536, (2048, 32), torch.uint8),
    ("down.global_scale", 1_769_480, 4, (), torch.float32),
    ("down.input_global_scale", 1_769_484, 4, (), torch.float32),
)


class ExpertPackStoreError(RuntimeError):
    pass


class ExpertPackIntegrityError(ExpertPackStoreError):
    pass


class ExpertPackFailedError(ExpertPackStoreError):
    pass


class ExpertCacheBusyError(ExpertPackStoreError):
    pass


@dataclass(frozen=True)
class ExpertOffloadConfig:
    manifest_path: str | os.PathLike[str]
    cache_ratio: float = 0.40
    cache_mib: int = 7168
    stage_slots: int = 16
    io_workers: int = 2
    stats_path: str | os.PathLike[str] | None = None
    mock_prefetch_log: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_path", Path(self.manifest_path).expanduser().resolve()
        )
        for name in ("stats_path", "mock_prefetch_log"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
        ratio = float(self.cache_ratio)
        if not math.isfinite(ratio) or not 8 / _NUM_EXPERTS <= ratio <= 1.0:
            raise ValueError("expert cache_ratio must be in [0.03125,1]")
        object.__setattr__(self, "cache_ratio", ratio)
        for value, label in (
            (self.cache_mib, "cache_mib"),
            (self.stage_slots, "stage_slots"),
            (self.io_workers, "io_workers"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"expert offload {label} must be a positive integer")
        if self.io_workers > self.stage_slots:
            raise ValueError("expert offload io_workers cannot exceed stage_slots")
        if self.cache_bytes > self.cache_mib * (1 << 20):
            raise ValueError(
                "layer-local expert cache exceeds --expert-cache-mib safety limit: "
                f"required={self.cache_bytes / (1 << 20):.3f} MiB, "
                f"limit={self.cache_mib} MiB"
            )

    @property
    def layer_capacity(self) -> int:
        return int(math.floor(_NUM_EXPERTS * self.cache_ratio))

    @property
    def realized_cache_ratio(self) -> float:
        return self.layer_capacity / _NUM_EXPERTS

    @property
    def scratch_capacity(self) -> int:
        return _NUM_EXPERTS - self.layer_capacity

    @property
    def cache_capacity(self) -> int:
        return len(_ACTIVE_LAYERS) * self.layer_capacity + self.scratch_capacity

    @property
    def cache_bytes(self) -> int:
        return self.cache_capacity * PAYLOAD_SIZE


@dataclass
class _PolicySlot:
    key: tuple[int, int] | None = None
    epoch: int = 0
    last_touch: int = 0
    leases: int = 0


@dataclass(frozen=True)
class _Placement:
    key: tuple[int, int]
    slot: int
    hit: bool
    prior_key: tuple[int, int] | None
    prior_epoch: int


class _CachePolicy:
    """Transactional strict-LRU metadata for one fixed slot interval."""

    def __init__(
        self, capacity: int, *, slot_offset: int = 0, layer_id: int | None = None
    ):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.slot_offset = slot_offset
        self.layer_id = layer_id
        self.slots = [_PolicySlot() for _ in range(capacity)]
        self.key_to_slot: dict[tuple[int, int], int] = {}
        self._clock = 0

    @property
    def resident(self) -> int:
        return len(self.key_to_slot)

    def _local(self, global_slot: int) -> int:
        return global_slot - self.slot_offset

    def plan(self, keys: Sequence[tuple[int, int]]) -> tuple[_Placement, ...]:
        if len(set(keys)) != len(keys):
            raise ValueError("cache placement keys must be unique")
        if self.layer_id is not None and any(key[0] != self.layer_id for key in keys):
            raise ValueError("layer-local cache received a key from another layer")
        protected = {self.key_to_slot[key] for key in keys if key in self.key_to_slot}
        reserved = set(protected)
        placements: list[_Placement] = []
        for key in keys:
            hit_slot = self.key_to_slot.get(key)
            if hit_slot is not None:
                slot = self.slots[self._local(hit_slot)]
                placements.append(_Placement(key, hit_slot, True, slot.key, slot.epoch))
                continue
            empty = next(
                (
                    self.slot_offset + index
                    for index, slot in enumerate(self.slots)
                    if slot.key is None and self.slot_offset + index not in reserved
                ),
                None,
            )
            if empty is None:
                candidates = [
                    (slot.last_touch, self.slot_offset + index)
                    for index, slot in enumerate(self.slots)
                    if self.slot_offset + index not in reserved and slot.leases == 0
                ]
                if not candidates:
                    raise ExpertCacheBusyError("layer cache has no unleased victim")
                empty = min(candidates)[1]
            reserved.add(empty)
            victim = self.slots[self._local(empty)]
            placements.append(_Placement(key, empty, False, victim.key, victim.epoch))
        return tuple(placements)

    def commit_and_lease(
        self,
        placements: Sequence[_Placement],
        *,
        lease: bool = True,
        touch_hits: bool = True,
    ) -> tuple[tuple[int, int], ...]:
        for placement in placements:
            slot = self.slots[self._local(placement.slot)]
            if slot.key != placement.prior_key or slot.epoch != placement.prior_epoch:
                raise RuntimeError("expert cache placement changed before commit")
            if not placement.hit and slot.leases:
                raise RuntimeError("expert cache victim became leased before commit")
            if placement.hit and self.key_to_slot.get(placement.key) != placement.slot:
                raise RuntimeError("expert cache hit disappeared before commit")
        result: list[tuple[int, int]] = []
        for placement in placements:
            slot = self.slots[self._local(placement.slot)]
            if not placement.hit:
                if slot.key is not None:
                    self.key_to_slot.pop(slot.key)
                slot.epoch += 1
                slot.key = placement.key
                self.key_to_slot[placement.key] = placement.slot
            if not placement.hit or touch_hits:
                self._clock += 1
                slot.last_touch = self._clock
            if lease:
                slot.leases += 1
            result.append((placement.slot, slot.epoch))
        return tuple(result)

    def release(self, assignments: Sequence[tuple[int, int]]) -> None:
        for global_slot, epoch in assignments:
            slot = self.slots[self._local(global_slot)]
            if slot.epoch != epoch or slot.key is None or slot.leases <= 0:
                raise RuntimeError(f"stale expert lease for slot {global_slot}")
        for global_slot, _ in assignments:
            self.slots[self._local(global_slot)].leases -= 1

    def clear(self) -> None:
        if any(slot.leases for slot in self.slots):
            raise ExpertCacheBusyError("cannot clear a cache with active leases")
        self.key_to_slot.clear()
        self._clock = 0
        for slot in self.slots:
            slot.key = None
            slot.epoch += 1
            slot.last_touch = 0


@dataclass
class _SlotRuntime:
    ready: torch.cuda.Event | None = None
    last_use: torch.cuda.Event | None = None


@dataclass(frozen=True)
class _ReadResult:
    read_ns: int
    checksum_ns: int


class PinnedExpertRepository:
    """All 9,728 payloads in final pinned host locations."""

    def __init__(
        self, manifest: Any, pack_path: Path, *, io_workers: int, queue_depth: int
    ):
        self.manifest = manifest
        self.pack_path = pack_path
        self.io_workers = io_workers
        self.queue_depth = queue_depth
        self.layers: dict[int, torch.Tensor] = {}
        self.reads = self.read_bytes = self.read_ns = self.checksum_ns = 0
        self._closed = False

    @staticmethod
    def _digest(record: Any) -> str:
        digest = getattr(record, "sha256", None)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("ExpertPack record has no valid SHA-256")
        return digest.lower()

    def _read_one(self, fd: int, layer: int, expert: int) -> _ReadResult:
        target = self.layers[layer][expert]
        record = self.manifest.record(layer, expert)
        if int(record.payload_size) != PAYLOAD_SIZE:
            raise ValueError("ExpertPack record payload size is invalid")
        view = memoryview(target.numpy()).cast("B")
        started = time.perf_counter_ns()
        read_bytes = os.preadv(fd, [view], int(record.offset))
        read_ns = time.perf_counter_ns() - started
        if read_bytes != PAYLOAD_SIZE:
            raise OSError(
                f"short ExpertPack read for layer={layer} expert={expert}: "
                f"{read_bytes} != {PAYLOAD_SIZE}"
            )
        started = time.perf_counter_ns()
        actual = hashlib.sha256(view).hexdigest()
        checksum_ns = time.perf_counter_ns() - started
        if actual != self._digest(record):
            raise ExpertPackIntegrityError(
                f"ExpertPack checksum mismatch for layer={layer} expert={expert}"
            )
        return _ReadResult(read_ns, checksum_ns)

    def _consume(self, result: _ReadResult) -> None:
        self.reads += 1
        self.read_bytes += PAYLOAD_SIZE
        self.read_ns += result.read_ns
        self.checksum_ns += result.checksum_ns

    def load(self) -> None:
        if self.layers:
            return
        try:
            for layer in _ACTIVE_LAYERS:
                self.layers[layer] = torch.empty(
                    (_NUM_EXPERTS, PAYLOAD_SIZE), dtype=torch.uint8, pin_memory=True
                )
        except BaseException:
            self.layers.clear()
            raise
        fd = os.open(self.pack_path, os.O_RDONLY)
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.io_workers, thread_name_prefix="qwen35-expert-load"
            ) as executor:
                pending: set[concurrent.futures.Future[_ReadResult]] = set()
                for layer in _ACTIVE_LAYERS:
                    for expert in range(_NUM_EXPERTS):
                        pending.add(executor.submit(self._read_one, fd, layer, expert))
                        if len(pending) >= self.queue_depth:
                            done, pending = concurrent.futures.wait(
                                pending, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for future in done:
                                self._consume(future.result())
                for future in concurrent.futures.as_completed(pending):
                    self._consume(future.result())
        except BaseException:
            self.layers.clear()
            raise
        finally:
            os.close(fd)

    def payload(self, layer: int, expert: int) -> torch.Tensor:
        if self._closed or layer not in self.layers:
            raise ExpertPackStoreError("pinned expert repository is unavailable")
        return self.layers[layer][expert]

    @property
    def nbytes(self) -> int:
        return len(_ACTIVE_LAYERS) * _NUM_EXPERTS * PAYLOAD_SIZE

    def close(self) -> None:
        self._closed = True
        self.layers.clear()


@dataclass
class _PrefetchBatch:
    command: PrefetchCommand
    candidates: tuple[int, ...]
    hit_experts: frozenset[int]
    copied_events: dict[int, torch.cuda.Event]
    trigger_event: torch.cuda.Event
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None
    actual: frozenset[int] = frozenset()
    deadline_event: torch.cuda.Event | None = None
    wait_end_event: torch.cuda.Event | None = None


class ExpertLease:
    def __init__(
        self,
        store: "ExpertPackStore",
        *,
        layer_id: int,
        expert_ids: tuple[int, ...],
        assignments: tuple[tuple[int, int], ...],
        scratch_slots: tuple[int, ...],
        gate_up: Weight,
        down: Weight,
        expert_to_slot: torch.Tensor,
    ) -> None:
        self._store = store
        self.layer_id = layer_id
        self.expert_ids = expert_ids
        self._assignments = assignments
        self._scratch_slots = scratch_slots
        self.gate_up = gate_up
        self.down = down
        self.expert_to_slot = expert_to_slot
        self.first_slot = assignments[0][0] if assignments else scratch_slots[0]
        self._released = False

    @property
    def slot_indices(self) -> tuple[int, ...]:
        return tuple(slot for slot, _ in self._assignments) + self._scratch_slots

    @property
    def released(self) -> bool:
        return self._released

    def release_after(self, stream: torch.cuda.Stream | None = None) -> None:
        if not self._released:
            self._store._release(
                self.layer_id, self._assignments, self._scratch_slots, stream
            )
            self._released = True

    def __enter__(self) -> "ExpertLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release_after()


class ExpertPackStore:
    """One-owner layer-local expert cache for Qwen3.5 layers 1..38."""

    def __init__(
        self,
        config: ExpertOffloadConfig | str | os.PathLike[str],
        *,
        device: torch.device | str | int | None = None,
    ) -> None:
        if not isinstance(config, ExpertOffloadConfig):
            config = ExpertOffloadConfig(config)
        if sys.byteorder != "little":
            raise RuntimeError("Qwen3.5 ExpertPack requires a little-endian host")
        self.config = config
        self.manifest = load_manifest(config.manifest_path, verify_identity=True)
        self.pack_path = Path(self.manifest.pack_path(config.manifest_path)).resolve()
        if not self.pack_path.is_file():
            raise FileNotFoundError(self.pack_path)
        if self.pack_path.stat().st_size != self.manifest.pack_size:
            raise ValueError("ExpertPack file size differs from manifest")
        self._lock = threading.RLock()
        self._state = "OPEN"
        self._failure: BaseException | None = None
        self._closed = False
        self._device: torch.device | None = None
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._demand_stream: torch.cuda.Stream | None = None
        self._repository = PinnedExpertRepository(
            self.manifest,
            self.pack_path,
            io_workers=config.io_workers,
            queue_depth=config.stage_slots,
        )
        self._policies = {
            layer: _CachePolicy(
                config.layer_capacity,
                slot_offset=(layer - 1) * config.layer_capacity,
                layer_id=layer,
            )
            for layer in _ACTIVE_LAYERS
        }
        scratch_start = len(_ACTIVE_LAYERS) * config.layer_capacity
        self._scratch_slots = tuple(
            range(scratch_start, scratch_start + config.scratch_capacity)
        )
        self._slot_runtime = [_SlotRuntime() for _ in range(config.cache_capacity)]
        self._cache_components: dict[str, torch.Tensor] = {}
        self._gate_up: Weight | None = None
        self._down: Weight | None = None
        self._request_epoch = 0
        self._cache_precleared = False
        self._active_replay: MockPrefetchConfig | None = None
        self._replay_origin: torch.cuda.Event | None = None
        self._prefetch_batches: dict[tuple[int, int], _PrefetchBatch] = {}
        self._last_replay_metrics: dict[str, Any] | None = None
        self._current_replay_metrics: dict[str, Any] | None = None
        self._route_calls_by_layer = {layer: 0 for layer in _ACTIVE_LAYERS}
        self._route_tokens_by_layer = {layer: 0 for layer in _ACTIVE_LAYERS}
        self._unique_experts_peak_by_layer = {layer: 0 for layer in _ACTIVE_LAYERS}
        self._stats: dict[str, int | float | str] = {
            "cache_capacity_experts": config.cache_capacity,
            "cache_capacity_bytes": config.cache_bytes,
            "cache_ratio_configured": config.cache_ratio,
            "cache_ratio_realized": config.realized_cache_ratio,
            "layer_cache_capacity_experts": config.layer_capacity,
            "prefill_scratch_capacity_experts": config.scratch_capacity,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "startup_pack_reads": 0,
            "startup_pack_read_bytes": 0,
            "runtime_pack_reads": 0,
            "h2d_bytes": 0,
            "fatal_errors": 0,
            "io_errors": 0,
            "checksum_errors": 0,
            "cuda_errors": 0,
            "epoch_errors": 0,
            "cache_policy": "layer-local-strict-lru-v1",
            "expert_source": "pinned-memory",
        }
        atexit.register(self.close)
        if device is not None:
            self.initialize_device_cache(device)

    @property
    def state(self) -> str:
        return self._state

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    @property
    def cache_capacity(self) -> int:
        return self.config.cache_capacity

    @property
    def cache_nbytes(self) -> int:
        return self.config.cache_bytes

    @property
    def request_epoch(self) -> int:
        return self._request_epoch

    @property
    def gate_up(self) -> Weight:
        if self._gate_up is None:
            raise RuntimeError("expert device cache is not initialized")
        return self._gate_up

    @property
    def down(self) -> Weight:
        if self._down is None:
            raise RuntimeError("expert device cache is not initialized")
        return self._down

    def _ensure_usable(self, *, require_device: bool = True) -> None:
        if self._closed:
            raise ExpertPackStoreError("expert pack store is closed")
        if self._state == "FAILED":
            error = ExpertPackFailedError(
                "expert pack store is FAILED and requires process restart"
            )
            if self._failure is not None:
                raise error from self._failure
            raise error
        if require_device and self._state != "READY":
            raise ExpertPackStoreError("expert device cache is not initialized")

    def _latch_failure(self, error: BaseException, category: str) -> None:
        if self._state == "FAILED":
            return
        self._failure = ExpertPackFailedError(f"{type(error).__name__}: {error}")
        self._state = "FAILED"
        self._stats["fatal_errors"] = int(self._stats["fatal_errors"]) + 1
        key = {
            "io": "io_errors",
            "checksum": "checksum_errors",
            "cuda": "cuda_errors",
            "epoch": "epoch_errors",
        }.get(category)
        if key:
            self._stats[key] = int(self._stats[key]) + 1
        logger.error("Qwen3.5 expert store entered FAILED: %s", error)

    def fail(self, error: BaseException, category: str = "cuda") -> None:
        with self._lock:
            if not self._closed:
                self._latch_failure(error, category)

    def initialize_device_cache(self, device: torch.device | str | int) -> None:
        with self._lock:
            self._ensure_usable(require_device=False)
            if not torch.cuda.is_available():
                raise RuntimeError("Qwen3.5 ExpertPack runtime requires CUDA")
            requested = torch.device(
                f"cuda:{device}" if isinstance(device, int) else device
            )
            index = (
                requested.index
                if requested.index is not None
                else torch.cuda.current_device()
            )
            normalized = torch.device("cuda", index)
            if self._device is not None:
                if self._device != normalized:
                    raise ValueError(
                        "expert cache is already initialized on another GPU"
                    )
                return
            try:
                self._repository.load()
                self._stats["startup_pack_reads"] = self._repository.reads
                self._stats["startup_pack_read_bytes"] = self._repository.read_bytes
                self._stats["startup_pack_read_ns"] = self._repository.read_ns
                self._stats["startup_checksum_ns"] = self._repository.checksum_ns
                with torch.cuda.device(normalized):
                    components = {
                        name: torch.empty(
                            (self.cache_capacity,) + shape,
                            dtype=dtype,
                            device=normalized,
                        )
                        for name, _, _, shape, dtype in _COMPONENT_LAYOUT
                    }
                    prefetch_stream = torch.cuda.Stream(device=normalized, priority=0)
                    demand_stream = torch.cuda.Stream(device=normalized, priority=-1)
                actual_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in components.values()
                )
                if actual_bytes != self.cache_nbytes:
                    raise RuntimeError("typed expert cache byte count is incorrect")
            except ExpertPackIntegrityError as error:
                self._latch_failure(error, "checksum")
                raise
            except OSError as error:
                self._latch_failure(error, "io")
                raise
            except BaseException as error:
                self._latch_failure(error, "cuda")
                raise
            self._device = normalized
            self._cache_components = components
            self._prefetch_stream = prefetch_stream
            self._demand_stream = demand_stream
            self._gate_up = Weight(
                "nvfp4",
                components["gate_up.data"],
                (self.cache_capacity, 1024, 2048),
                components["gate_up.block_scale"],
                components["gate_up.global_scale"],
                components["gate_up.input_global_scale"],
            )
            self._down = Weight(
                "nvfp4",
                components["down.data"],
                (self.cache_capacity, 2048, 512),
                components["down.block_scale"],
                components["down.global_scale"],
                components["down.input_global_scale"],
            )
            self._state = "READY"
            logger.info(
                "Qwen3.5 pinned ExpertPack ready: ratio=%.6f layer=%d scratch=%d "
                "cache=%.3f MiB host=%.3f GiB",
                self.config.realized_cache_ratio,
                self.config.layer_capacity,
                self.config.scratch_capacity,
                self.cache_nbytes / (1 << 20),
                self._repository.nbytes / (1 << 30),
            )

    @staticmethod
    def _host_component(
        payload: torch.Tensor,
        offset: int,
        nbytes: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = payload.narrow(0, offset, nbytes)
        if dtype == torch.float32:
            value = value.view(torch.float32)
        return value.reshape(shape or ())

    def _enqueue_install(
        self,
        payload: torch.Tensor,
        slot_index: int,
        stream: torch.cuda.Stream,
        *,
        trigger_event: torch.cuda.Event | None = None,
        timing: bool = False,
    ) -> torch.cuda.Event:
        runtime = self._slot_runtime[slot_index]
        with torch.cuda.stream(stream):
            if trigger_event is not None:
                stream.wait_event(trigger_event)
            if runtime.last_use is not None:
                stream.wait_event(runtime.last_use)
            elif runtime.ready is not None:
                stream.wait_event(runtime.ready)
            for name, offset, nbytes, shape, dtype in _COMPONENT_LAYOUT:
                source = self._host_component(payload, offset, nbytes, shape, dtype)
                self._cache_components[name][slot_index].copy_(
                    source, non_blocking=True
                )
            ready = torch.cuda.Event(enable_timing=timing)
            ready.record(stream)
        return ready

    @staticmethod
    def _ordered_unique(values: Iterable[int]) -> tuple[int, ...]:
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _final_lru(values: Sequence[int], capacity: int) -> tuple[int, ...]:
        newest_first = tuple(dict.fromkeys(reversed(values)))[:capacity]
        return tuple(reversed(newest_first))

    def _install_placements(
        self,
        placements: Sequence[_Placement],
        *,
        stream: torch.cuda.Stream,
        timing: bool = False,
    ) -> dict[int, torch.cuda.Event]:
        ready: dict[int, torch.cuda.Event] = {}
        for placement in placements:
            if placement.hit:
                continue
            event = self._enqueue_install(
                self._repository.payload(*placement.key),
                placement.slot,
                stream,
                timing=timing,
            )
            ready[placement.slot] = event
            self._stats["h2d_bytes"] = int(self._stats["h2d_bytes"]) + PAYLOAD_SIZE
        return ready

    def prefetch(
        self, command: PrefetchCommand, trigger_event: torch.cuda.Event
    ) -> None:
        """Enqueue predicted H2D work without synchronizing the host or GPU."""
        with self._lock:
            self._ensure_usable()
            if self._active_replay is None:
                raise RuntimeError("prefetch requires an active replay")
            if command.request_epoch != self._request_epoch:
                error = ExpertPackFailedError("stale prefetch request epoch")
                self._latch_failure(error, "epoch")
                raise error
            if command.target_layer not in _ACTIVE_LAYERS:
                raise ValueError("prefetch target layer must be in [1,38]")
            candidates = tuple(command.candidate_expert_ids)
            if len(candidates) > self.config.layer_capacity:
                raise ValueError("prefetch top_k exceeds the layer cache capacity")
            if len(set(candidates)) != len(candidates) or any(
                expert < 0 or expert >= _NUM_EXPERTS for expert in candidates
            ):
                raise ValueError("prefetch candidates must be unique IDs in [0,256)")
            key = (command.output_row, command.target_layer)
            if key in self._prefetch_batches:
                raise RuntimeError("prefetch command was issued twice")
            policy = self._policies[command.target_layer]
            placements = policy.plan(
                tuple((command.target_layer, expert) for expert in candidates)
            )
            misses = tuple(item for item in placements if not item.hit)
            assert self._prefetch_stream is not None
            start_event = end_event = None
            copied: dict[int, torch.cuda.Event] = {}
            try:
                if misses:
                    with torch.cuda.stream(self._prefetch_stream):
                        self._prefetch_stream.wait_event(trigger_event)
                        start_event = torch.cuda.Event(enable_timing=True)
                        start_event.record(self._prefetch_stream)
                    ready_by_slot = self._install_placements(
                        misses, stream=self._prefetch_stream, timing=True
                    )
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record(self._prefetch_stream)
                else:
                    ready_by_slot = {}
                policy.commit_and_lease(placements, lease=False, touch_hits=False)
                for placement in misses:
                    runtime = self._slot_runtime[placement.slot]
                    runtime.ready = ready_by_slot[placement.slot]
                    runtime.last_use = None
                    copied[placement.key[1]] = ready_by_slot[placement.slot]
            except BaseException as error:
                self._latch_failure(error, "cuda")
                raise
            self._prefetch_batches[key] = _PrefetchBatch(
                command=command,
                candidates=candidates,
                hit_experts=frozenset(item.key[1] for item in placements if item.hit),
                copied_events=copied,
                trigger_event=trigger_event,
                start_event=start_event,
                end_event=end_event,
            )
            replay = self._replay_counters()
            replay["prediction_candidates"] += len(candidates)
            replay["prefetch_cache_hits"] += len(placements) - len(misses)
            replay["prefetch_h2d_issued"] += len(misses)
            replay["prefetch_h2d_bytes"] += len(misses) * PAYLOAD_SIZE
            replay["prefetch_evictions"] += sum(
                not item.hit and item.prior_key is not None for item in placements
            )

    def acquire(
        self,
        layer_id: int,
        expert_ids: torch.Tensor,
        *,
        output_row: int | None = None,
    ) -> ExpertLease:
        """Acquire one routed working set from pinned host memory or GPU cache."""
        if layer_id not in _ACTIVE_LAYERS:
            raise ValueError("expert offload layer_id must be in 1..38")
        if not isinstance(expert_ids, torch.Tensor):
            raise TypeError("expert_ids must be a torch.Tensor")
        if expert_ids.ndim != 2 or expert_ids.shape[1] != 8 or expert_ids.numel() == 0:
            raise ValueError("expert_ids must be a nonempty [T,8] tensor")
        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("expert_ids must use int32 or int64")
        with self._lock:
            self._ensure_usable()
        try:
            route_ids = tuple(
                int(value) for value in expert_ids.detach().cpu().reshape(-1).tolist()
            )
        except BaseException as error:
            with self._lock:
                self._latch_failure(error, "cuda")
            raise
        if any(expert < 0 or expert >= _NUM_EXPERTS for expert in route_ids):
            raise ValueError("expert_ids contains an out-of-range expert")
        unique = self._ordered_unique(route_ids)
        with self._lock:
            self._ensure_usable()
            policy = self._policies[layer_id]
            retained = self._final_lru(route_ids, self.config.layer_capacity)
            retained_set = set(retained)
            scratch_experts = tuple(
                expert for expert in unique if expert not in retained_set
            )
            if len(scratch_experts) > len(self._scratch_slots):
                raise ExpertCacheBusyError("prefill scratch capacity is insufficient")
            placements = policy.plan(tuple((layer_id, expert) for expert in retained))
            misses = tuple(item for item in placements if not item.hit)
            batch = (
                self._prefetch_batches.get((output_row, layer_id))
                if output_row is not None and expert_ids.shape[0] == 1
                else None
            )
            assert self._device is not None
            current_stream = torch.cuda.current_stream(self._device)
            if batch is not None:
                deadline = torch.cuda.Event(enable_timing=True)
                deadline.record(current_stream)
                batch.actual = frozenset(unique)
                batch.deadline_event = deadline
            assert self._demand_stream is not None
            try:
                ready_by_slot = self._install_placements(
                    misses,
                    stream=self._demand_stream,
                    timing=self._active_replay is not None,
                )
                scratch_ready: dict[int, torch.cuda.Event] = {}
                used_scratch = self._scratch_slots[: len(scratch_experts)]
                for slot, expert in zip(used_scratch, scratch_experts):
                    ready = self._enqueue_install(
                        self._repository.payload(layer_id, expert),
                        slot,
                        self._demand_stream,
                        timing=self._active_replay is not None,
                    )
                    scratch_ready[slot] = ready
                    self._stats["h2d_bytes"] = (
                        int(self._stats["h2d_bytes"]) + PAYLOAD_SIZE
                    )
                assignments = policy.commit_and_lease(
                    placements, lease=True, touch_hits=True
                )
                for placement in misses:
                    runtime = self._slot_runtime[placement.slot]
                    runtime.ready = ready_by_slot[placement.slot]
                    runtime.last_use = None
                for slot, ready in scratch_ready.items():
                    runtime = self._slot_runtime[slot]
                    runtime.ready = ready
                    runtime.last_use = None
                for slot, _ in assignments:
                    ready = self._slot_runtime[slot].ready
                    if ready is not None:
                        current_stream.wait_event(ready)
                for ready in scratch_ready.values():
                    current_stream.wait_event(ready)
                if batch is not None:
                    wait_end = torch.cuda.Event(enable_timing=True)
                    wait_end.record(current_stream)
                    batch.wait_end_event = wait_end
                mapping_cpu = torch.full((_NUM_EXPERTS,), -1, dtype=torch.int32)
                for placement, assignment in zip(placements, assignments):
                    mapping_cpu[placement.key[1]] = assignment[0]
                for slot, expert in zip(used_scratch, scratch_experts):
                    mapping_cpu[expert] = slot
                expert_to_slot = mapping_cpu.to(self._device, non_blocking=False)
            except BaseException as error:
                self._latch_failure(error, "cuda")
                raise
            hits = len(placements) - len(misses)
            evictions = sum(
                not item.hit and item.prior_key is not None for item in placements
            )
            self._stats["cache_hits"] = int(self._stats["cache_hits"]) + hits
            self._stats["cache_misses"] = (
                int(self._stats["cache_misses"]) + len(misses) + len(scratch_experts)
            )
            self._stats["cache_evictions"] = (
                int(self._stats["cache_evictions"]) + evictions
            )
            self._route_calls_by_layer[layer_id] += 1
            self._route_tokens_by_layer[layer_id] += int(expert_ids.shape[0])
            self._unique_experts_peak_by_layer[layer_id] = max(
                self._unique_experts_peak_by_layer[layer_id], len(unique)
            )
            if self._active_replay is not None:
                replay = self._replay_counters()
                if expert_ids.shape[0] > 1:
                    replay["prefill_scratch_peak"] = max(
                        replay["prefill_scratch_peak"], len(scratch_experts)
                    )
                else:
                    predicted = set(batch.candidates) if batch is not None else set()
                    useful = predicted.intersection(unique)
                    replay["prefetch_useful"] += len(useful)
                    replay["prefetch_wasted"] += len(predicted - set(unique))
                    replay["demand_prefetch_hits"] += len(useful)
                    replay["demand_hot_hits"] += max(0, hits - len(useful))
                    replay["demand_misses"] += len(misses)
                    replay["demand_h2d_bytes"] += len(misses) * PAYLOAD_SIZE
                    replay["demand_evictions"] += evictions
            return ExpertLease(
                self,
                layer_id=layer_id,
                expert_ids=unique,
                assignments=assignments,
                scratch_slots=tuple(used_scratch),
                gate_up=self.gate_up,
                down=self.down,
                expert_to_slot=expert_to_slot,
            )

    def _release(
        self,
        layer_id: int,
        assignments: Sequence[tuple[int, int]],
        scratch_slots: Sequence[int],
        stream: torch.cuda.Stream | None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            if self._device is None:
                raise ExpertPackStoreError("expert device cache is not initialized")
            stream = stream or torch.cuda.current_stream(self._device)
            event = torch.cuda.Event(enable_timing=False)
            event.record(stream)
            self._policies[layer_id].release(assignments)
            for slot, _ in assignments:
                self._slot_runtime[slot].last_use = event
            for slot in scratch_slots:
                self._slot_runtime[slot].last_use = event

    def _replay_counters(self) -> dict[str, Any]:
        if self._current_replay_metrics is None:
            raise RuntimeError("no active replay metrics")
        return self._current_replay_metrics

    def clear_cache_after_record(self) -> int:
        """Synchronize and invalidate cache state outside the measured replay."""
        with self._lock:
            self._ensure_usable()
            assert self._device is not None
            torch.cuda.synchronize(self._device)
            for policy in self._policies.values():
                policy.clear()
            for runtime in self._slot_runtime:
                runtime.ready = None
                runtime.last_use = None
            self._request_epoch += 1
            self._cache_precleared = True
            return self._request_epoch

    def begin_replay(self, config: MockPrefetchConfig) -> int:
        with self._lock:
            self._ensure_usable()
            if config.top_k > self.config.layer_capacity:
                raise ValueError(
                    f"mock prefetch top_k={config.top_k} exceeds per-layer "
                    f"capacity={self.config.layer_capacity}"
                )
            if not self._cache_precleared:
                raise RuntimeError("mock replay requires a cache cleared after record")
            self._cache_precleared = False
            self._active_replay = config
            self._prefetch_batches.clear()
            self._last_replay_metrics = None
            self._current_replay_metrics = {
                "request_epoch": self._request_epoch,
                "cache_ratio_configured": self.config.cache_ratio,
                "cache_ratio_realized": self.config.realized_cache_ratio,
                "layer_capacity": self.config.layer_capacity,
                "scratch_capacity": self.config.scratch_capacity,
                "prediction_candidates": 0,
                "prefetch_cache_hits": 0,
                "prefetch_h2d_issued": 0,
                "prefetch_h2d_bytes": 0,
                "prefetch_useful": 0,
                "prefetch_on_time": 0,
                "prefetch_late": 0,
                "prefetch_wasted": 0,
                "prefetch_evictions": 0,
                "demand_hot_hits": 0,
                "demand_prefetch_hits": 0,
                "demand_misses": 0,
                "demand_h2d_bytes": 0,
                "demand_evictions": 0,
                "prefill_scratch_peak": 0,
                "prefetch_h2d_ms": 0.0,
                "available_window_ms": 0.0,
                "exposed_stall_ms": 0.0,
                "hidden_h2d_ms": 0.0,
                "runtime_pack_reads": 0,
            }
            origin = torch.cuda.Event(enable_timing=True)
            origin.record(torch.cuda.current_stream(self._device))
            self._replay_origin = origin
            return self._request_epoch

    def reset_for_replay(self, config: MockPrefetchConfig) -> int:
        """Compatibility helper used by direct callers and tests."""
        self.clear_cache_after_record()
        return self.begin_replay(config)

    def finish_replay(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_usable()
            if self._active_replay is None or self._replay_origin is None:
                raise RuntimeError("no active replay")
            assert self._device is not None
            torch.cuda.synchronize(self._device)
            raw: list[dict[str, Any]] = []
            counters = self._replay_counters()
            for batch in self._prefetch_batches.values():
                useful = set(batch.candidates).intersection(batch.actual)
                on_time = late = 0
                trigger_ms = self._replay_origin.elapsed_time(batch.trigger_event)
                deadline_ms = (
                    self._replay_origin.elapsed_time(batch.deadline_event)
                    if batch.deadline_event is not None
                    else None
                )
                wait_end_ms = (
                    self._replay_origin.elapsed_time(batch.wait_end_event)
                    if batch.wait_end_event is not None
                    else None
                )
                for expert in useful:
                    ready = batch.copied_events.get(expert)
                    if ready is None:
                        on_time += 1
                    elif (
                        deadline_ms is not None
                        and self._replay_origin.elapsed_time(ready) <= deadline_ms
                    ):
                        on_time += 1
                    else:
                        late += 1
                counters["prefetch_on_time"] += on_time
                counters["prefetch_late"] += late
                h2d_ms = (
                    batch.start_event.elapsed_time(batch.end_event)
                    if batch.start_event is not None and batch.end_event is not None
                    else 0.0
                )
                window_ms = (
                    max(0.0, deadline_ms - trigger_ms)
                    if deadline_ms is not None
                    else 0.0
                )
                stall_ms = (
                    max(0.0, wait_end_ms - deadline_ms)
                    if deadline_ms is not None and wait_end_ms is not None
                    else 0.0
                )
                hidden_ms = max(0.0, min(h2d_ms, h2d_ms - stall_ms))
                counters["prefetch_h2d_ms"] += h2d_ms
                counters["available_window_ms"] += window_ms
                counters["exposed_stall_ms"] += stall_ms
                counters["hidden_h2d_ms"] += hidden_ms
                raw.append(
                    {
                        "request_epoch": self._request_epoch,
                        "output_row": batch.command.output_row,
                        "trigger_layer": batch.command.trigger_layer,
                        "target_layer": batch.command.target_layer,
                        "candidates": list(batch.candidates),
                        "actual": sorted(batch.actual),
                        "h2d_experts": len(batch.copied_events),
                        "useful": len(useful),
                        "on_time": on_time,
                        "late": late,
                        "h2d_ms": h2d_ms,
                        "available_window_ms": window_ms,
                        "exposed_stall_ms": stall_ms,
                    }
                )
            h2d_ms = float(counters["prefetch_h2d_ms"])
            counters["overlap_ratio"] = (
                min(
                    1.0,
                    max(0.0, float(counters["hidden_h2d_ms"]) / h2d_ms),
                )
                if h2d_ms
                else 0.0
            )
            counters["resident_by_layer"] = {
                str(layer): policy.resident for layer, policy in self._policies.items()
            }
            result = dict(counters)
            self._last_replay_metrics = dict(result)
            self._active_replay = None
            self._current_replay_metrics = None
            self._prefetch_batches.clear()
            self._replay_origin = None
            if self.config.mock_prefetch_log is not None:
                path = Path(self.config.mock_prefetch_log)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as output:
                    output.write(
                        json.dumps({**result, "commands": raw}, sort_keys=True) + "\n"
                    )
                    output.flush()
                    os.fsync(output.fileno())
            return result

    def abort_replay(self) -> None:
        with self._lock:
            self._active_replay = None
            self._current_replay_metrics = None
            self._prefetch_batches.clear()
            self._replay_origin = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            value: dict[str, Any] = dict(self._stats)
            resident_by_layer = {
                layer: policy.resident for layer, policy in self._policies.items()
            }
            value.update(
                {
                    "state": self._state,
                    "failure": repr(self._failure) if self._failure else None,
                    "manifest_path": str(self.config.manifest_path),
                    "pack_path": str(self.pack_path),
                    "pack_fd_open": False,
                    "pinned_repository_bytes": self._repository.nbytes,
                    "resident_experts": sum(resident_by_layer.values()),
                    "resident_bytes": sum(resident_by_layer.values()) * PAYLOAD_SIZE,
                    "resident_by_layer": resident_by_layer,
                    "route_calls_by_layer": dict(self._route_calls_by_layer),
                    "route_tokens_by_layer": dict(self._route_tokens_by_layer),
                    "unique_experts_peak_by_layer": dict(
                        self._unique_experts_peak_by_layer
                    ),
                    "request_epoch": self._request_epoch,
                    "last_mock_prefetch": self._last_replay_metrics,
                }
            )
            return value

    def _write_stats(self) -> None:
        if self.config.stats_path is None:
            return
        path = Path(self.config.stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._device is not None:
                    torch.cuda.synchronize(self._device)
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
            try:
                self._write_stats()
            except Exception:
                logger.exception("failed to write Qwen3.5 expert statistics")
            self._repository.close()
            self._cache_components.clear()
            self._gate_up = None
            self._down = None
            self._state = "CLOSED"
            try:
                atexit.unregister(self.close)
            except Exception:
                pass

    def __enter__(self) -> "ExpertPackStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "ExpertCacheBusyError",
    "ExpertLease",
    "ExpertOffloadConfig",
    "ExpertPackFailedError",
    "ExpertPackIntegrityError",
    "ExpertPackStore",
    "ExpertPackStoreError",
    "PinnedExpertRepository",
]
