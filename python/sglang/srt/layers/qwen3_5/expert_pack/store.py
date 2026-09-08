# SPDX-License-Identifier: Apache-2.0
"""CUDA cache and lease management for the Qwen3.5 NVFP4 ExpertPack.

The file format lives in :mod:`.format`.  This module owns the runtime side of
the contract: bounded positional reads into pinned host buffers, per-record
integrity checks, a typed structure-of-arrays CUDA cache, and generation-aware
leases which make cache-slot reuse safe across CUDA streams.

The store deliberately serializes ``acquire``.  The first supported runner has
one owner and one compute stream; serializing metadata updates keeps the point
at which a new mapping becomes visible unambiguous while the file reads still
run concurrently in the configured I/O pool.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from ..weights import Weight
from .format import PAYLOAD_SIZE, load_manifest

logger = logging.getLogger(__name__)

_NUM_EXPERTS = 256
_ACTIVE_LAYERS = frozenset(range(1, 39))

# These offsets are part of SGLANG-QWEN35-NVFP4-EXPERTPACK-v1.  Keeping the
# runtime description explicit also makes construction of the typed SoA cache
# independent of JSON dictionary ordering.
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
    """Base class for ExpertPack runtime errors."""


class ExpertPackIntegrityError(ExpertPackStoreError):
    """A record read successfully but did not match its manifest digest."""


class ExpertPackFailedError(ExpertPackStoreError):
    """The process-local store has latched a fatal error and cannot recover."""


class ExpertCacheBusyError(ExpertPackStoreError):
    """No cache slot can be reused because every candidate has an active lease."""


@dataclass(frozen=True)
class ExpertOffloadConfig:
    """Public single-GPU ExpertPack runtime configuration."""

    manifest_path: str | os.PathLike[str]
    cache_mib: int = 7168
    stage_slots: int = 16
    io_workers: int = 2
    stats_path: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        manifest_path = Path(self.manifest_path).expanduser().resolve()
        object.__setattr__(self, "manifest_path", manifest_path)
        if self.stats_path is not None:
            object.__setattr__(
                self, "stats_path", Path(self.stats_path).expanduser().resolve()
            )
        for value, label in (
            (self.cache_mib, "cache_mib"),
            (self.stage_slots, "stage_slots"),
            (self.io_workers, "io_workers"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"expert offload {label} must be a positive integer")
        if self.cache_capacity < _NUM_EXPERTS:
            minimum_mib = (PAYLOAD_SIZE * _NUM_EXPERTS + (1 << 20) - 1) // (1 << 20)
            raise ValueError(
                "expert cache cannot hold one worst-case layer working set: "
                f"capacity={self.cache_capacity}, required={_NUM_EXPERTS}, "
                f"minimum={minimum_mib} MiB"
            )
        if self.io_workers > self.stage_slots:
            raise ValueError("expert offload io_workers cannot exceed stage_slots")

    @property
    def cache_capacity(self) -> int:
        return (self.cache_mib * (1 << 20)) // PAYLOAD_SIZE

    @property
    def cache_bytes(self) -> int:
        return self.cache_capacity * PAYLOAD_SIZE


@dataclass
class _PolicySlot:
    key: tuple[int, int] | None = None
    epoch: int = 0
    frequency: int = 0
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
    """Pure-Python LFU/LRU cache metadata with transactional placement.

    ``plan`` never changes visible mappings.  A caller can therefore perform
    reads and enqueue H2D copies, then call ``commit_and_lease`` only after all
    of those operations have succeeded.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.slots = [_PolicySlot() for _ in range(capacity)]
        self.key_to_slot: dict[tuple[int, int], int] = {}
        self.key_frequency: dict[tuple[int, int], int] = {}
        self._clock = 0

    @property
    def resident(self) -> int:
        return len(self.key_to_slot)

    def plan(self, keys: Sequence[tuple[int, int]]) -> tuple[_Placement, ...]:
        if len(set(keys)) != len(keys):
            raise ValueError("cache placement keys must be unique")
        protected = {self.key_to_slot[key] for key in keys if key in self.key_to_slot}
        reserved: set[int] = set(protected)
        placements: list[_Placement] = []
        for key in keys:
            hit_slot = self.key_to_slot.get(key)
            if hit_slot is not None:
                slot = self.slots[hit_slot]
                placements.append(_Placement(key, hit_slot, True, slot.key, slot.epoch))
                continue

            empty = next(
                (
                    index
                    for index, slot in enumerate(self.slots)
                    if slot.key is None and index not in reserved
                ),
                None,
            )
            if empty is None:
                candidates = [
                    (slot.frequency, slot.last_touch, index)
                    for index, slot in enumerate(self.slots)
                    if index not in reserved and slot.leases == 0
                ]
                if not candidates:
                    raise ExpertCacheBusyError(
                        "expert cache has no unleased slot for the requested working set"
                    )
                empty = min(candidates)[2]
            reserved.add(empty)
            victim = self.slots[empty]
            placements.append(_Placement(key, empty, False, victim.key, victim.epoch))
        return tuple(placements)

    def commit_and_lease(
        self, placements: Sequence[_Placement]
    ) -> tuple[tuple[int, int], ...]:
        """Atomically publish placements and return ``(slot, epoch)`` leases."""
        # First verify the complete transaction against the current state.
        for placement in placements:
            slot = self.slots[placement.slot]
            if slot.key != placement.prior_key or slot.epoch != placement.prior_epoch:
                raise RuntimeError("expert cache placement changed before commit")
            if not placement.hit and slot.leases:
                raise RuntimeError("expert cache victim became leased before commit")
            if placement.hit and self.key_to_slot.get(placement.key) != placement.slot:
                raise RuntimeError("expert cache hit disappeared before commit")

        result: list[tuple[int, int]] = []
        for placement in placements:
            slot = self.slots[placement.slot]
            if not placement.hit:
                if slot.key is not None:
                    self.key_to_slot.pop(slot.key)
                slot.epoch += 1
                slot.key = placement.key
                self.key_to_slot[placement.key] = placement.slot
            self._clock += 1
            frequency = self.key_frequency.get(placement.key, 0) + 1
            self.key_frequency[placement.key] = frequency
            slot.frequency = frequency
            slot.last_touch = self._clock
            slot.leases += 1
            result.append((placement.slot, slot.epoch))
        return tuple(result)

    def release(self, assignments: Sequence[tuple[int, int]]) -> None:
        for slot_index, epoch in assignments:
            slot = self.slots[slot_index]
            if slot.epoch != epoch or slot.key is None:
                raise RuntimeError(
                    f"stale expert lease for slot {slot_index}: "
                    f"lease epoch={epoch}, current epoch={slot.epoch}"
                )
            if slot.leases <= 0:
                raise RuntimeError(f"expert cache slot {slot_index} is not leased")
        for slot_index, _ in assignments:
            self.slots[slot_index].leases -= 1


@dataclass
class _SlotRuntime:
    ready: torch.cuda.Event | None = None
    last_use: torch.cuda.Event | None = None


@dataclass(frozen=True)
class _ReadResult:
    read_ns: int
    checksum_ns: int


class ExpertLease:
    """A complete layer working set held until its compute stream is recorded."""

    def __init__(
        self,
        store: "ExpertPackStore",
        *,
        layer_id: int,
        expert_ids: tuple[int, ...],
        assignments: tuple[tuple[int, int], ...],
        gate_up: Weight,
        down: Weight,
        expert_to_slot: torch.Tensor,
    ) -> None:
        self._store = store
        self.layer_id = layer_id
        self.expert_ids = expert_ids
        self._assignments = assignments
        self.gate_up = gate_up
        self.down = down
        self.expert_to_slot = expert_to_slot
        self.first_slot = assignments[0][0]
        self._released = False

    @property
    def slot_indices(self) -> tuple[int, ...]:
        return tuple(slot for slot, _ in self._assignments)

    @property
    def released(self) -> bool:
        return self._released

    def release_after(self, stream: torch.cuda.Stream | None = None) -> None:
        """Release after all cache consumers have been enqueued on ``stream``.

        Recording an event rather than synchronizing preserves overlap.  A
        later H2D overwrite waits for this event on the transfer stream.
        """
        if self._released:
            return
        self._store._release(self._assignments, stream)
        self._released = True

    def __enter__(self) -> "ExpertLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release_after()


class ExpertPackStore:
    """One-owner runtime cache for Qwen3.5 layers 1..38 ExpertPack records."""

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
        actual_pack_size = self.pack_path.stat().st_size
        if actual_pack_size != self.manifest.pack_size:
            raise ValueError(
                f"ExpertPack file size {actual_pack_size} != manifest size "
                f"{self.manifest.pack_size}"
            )

        self._fd = os.open(self.pack_path, os.O_RDONLY)
        self._lock = threading.RLock()
        self._state = "OPEN"
        self._failure: BaseException | None = None
        self._closed = False
        self._device: torch.device | None = None
        self._transfer_stream: torch.cuda.Stream | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._policy = _CachePolicy(config.cache_capacity)
        self._slot_runtime = [_SlotRuntime() for _ in range(config.cache_capacity)]
        self._staging: list[torch.Tensor] = []
        self._staging_ready: list[torch.cuda.Event | None] = []
        self._stage_cursor = 0
        self._cache_components: dict[str, torch.Tensor] = {}
        self._gate_up: Weight | None = None
        self._down: Weight | None = None
        self._route_calls_by_layer = {layer: 0 for layer in _ACTIVE_LAYERS}
        self._route_tokens_by_layer = {layer: 0 for layer in _ACTIVE_LAYERS}
        self._unique_experts_peak_by_layer = {layer: 0 for layer in _ACTIVE_LAYERS}
        self._stats: dict[str, int | str] = {
            "cache_capacity_experts": config.cache_capacity,
            "cache_capacity_bytes": config.cache_bytes,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "pack_reads": 0,
            "pack_read_bytes": 0,
            "pack_read_ns": 0,
            "checksum_ns": 0,
            "h2d_bytes": 0,
            "stage_wait_ns": 0,
            "io_errors": 0,
            "checksum_errors": 0,
            "cuda_errors": 0,
            "epoch_errors": 0,
            "fatal_errors": 0,
            "cache_policy": "lfu-lru-v1",
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
        if self._state != "FAILED":
            # Keeping the original exception would retain its traceback and,
            # for allocation failures, potentially several GiB of CUDA tensor
            # locals.  Preserve the diagnostic without retaining that frame.
            self._failure = ExpertPackFailedError(f"{type(error).__name__}: {error}")
            self._state = "FAILED"
            self._stats["fatal_errors"] = int(self._stats["fatal_errors"]) + 1
            key = {
                "io": "io_errors",
                "checksum": "checksum_errors",
                "cuda": "cuda_errors",
                "epoch": "epoch_errors",
            }.get(category)
            if key is not None:
                self._stats[key] = int(self._stats[key]) + 1
            logger.error(
                "Qwen3.5 ExpertPack store entered FAILED state: %s",
                error,
                exc_info=(
                    (type(error), error, error.__traceback__)
                    if error.__traceback__ is not None
                    else None
                ),
            )

    def fail(self, error: BaseException, category: str = "cuda") -> None:
        """Latch a fatal error detected outside the store's synchronous calls.

        CUDA work is asynchronous, so a runner may first observe an H2D or
        kernel failure at its final ``torch.cuda.synchronize``.  Calling this
        method makes that failure visible to health checks and prevents every
        later acquire in the same process.
        """
        if not isinstance(error, BaseException):
            raise TypeError("expert store failure must be an exception")
        if category not in {"cuda", "io", "checksum", "epoch"}:
            raise ValueError(f"unsupported expert store failure category: {category}")
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
            if requested.type != "cuda":
                raise ValueError("Qwen3.5 ExpertPack cache device must be CUDA")
            index = requested.index
            if index is None:
                index = torch.cuda.current_device()
            normalized = torch.device("cuda", index)
            if self._device is not None:
                if self._device != normalized:
                    raise ValueError(
                        f"expert cache already initialized on {self._device}, not {normalized}"
                    )
                return
            try:
                with torch.cuda.device(normalized):
                    components: dict[str, torch.Tensor] = {}
                    capacity = self.cache_capacity
                    for name, _, _, shape, dtype in _COMPONENT_LAYOUT:
                        components[name] = torch.empty(
                            (capacity,) + shape,
                            dtype=dtype,
                            device=normalized,
                        )
                    gate_up = Weight(
                        "nvfp4",
                        components["gate_up.data"],
                        (capacity, 1024, 2048),
                        components["gate_up.block_scale"],
                        components["gate_up.global_scale"],
                        components["gate_up.input_global_scale"],
                    )
                    down = Weight(
                        "nvfp4",
                        components["down.data"],
                        (capacity, 2048, 512),
                        components["down.block_scale"],
                        components["down.global_scale"],
                        components["down.input_global_scale"],
                    )
                    staging = [
                        torch.empty(PAYLOAD_SIZE, dtype=torch.uint8, pin_memory=True)
                        for _ in range(self.config.stage_slots)
                    ]
                    transfer_stream = torch.cuda.Stream(device=normalized)
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.io_workers,
                    thread_name_prefix="qwen35-expert-read",
                )
            except Exception as error:
                self._latch_failure(error, "cuda")
                raise

            actual_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in components.values()
            )
            if actual_bytes != self.cache_nbytes:
                error = RuntimeError(
                    f"typed expert cache bytes {actual_bytes} != budgeted {self.cache_nbytes}"
                )
                executor.shutdown(wait=True, cancel_futures=True)
                self._latch_failure(error, "cuda")
                raise error
            self._device = normalized
            self._cache_components = components
            self._gate_up = gate_up
            self._down = down
            self._staging = staging
            self._staging_ready = [None] * len(staging)
            self._transfer_stream = transfer_stream
            self._executor = executor
            self._state = "READY"
            self._stats["staging_slots"] = len(staging)
            self._stats["staging_bytes"] = len(staging) * PAYLOAD_SIZE
            self._stats["io_workers"] = self.config.io_workers
            logger.info(
                "Qwen3.5 ExpertPack ready: capacity=%d cache_bytes=%d staging=%d",
                self.cache_capacity,
                self.cache_nbytes,
                len(staging),
            )

    @staticmethod
    def _record_digest(record: Any) -> str:
        value = getattr(record, "sha256", None)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("expert pack record has no valid SHA-256")
        return value.lower()

    def _read_payload(self, record: Any, staging: torch.Tensor) -> _ReadResult:
        if staging.device.type != "cpu" or staging.dtype != torch.uint8:
            raise TypeError("expert staging buffer must be a CPU uint8 tensor")
        if staging.numel() != PAYLOAD_SIZE or not staging.is_contiguous():
            raise ValueError("expert staging buffer has the wrong payload size")
        offset = int(record.offset)
        length = int(record.payload_size)
        if offset < 0 or length != PAYLOAD_SIZE:
            raise ValueError("expert pack record has invalid bounds")
        view = memoryview(staging.numpy()).cast("B")
        started = time.perf_counter_ns()
        try:
            read_bytes = os.preadv(self._fd, [view], offset)
        except OSError:
            raise
        read_ns = time.perf_counter_ns() - started
        if read_bytes != length:
            raise OSError(
                f"short ExpertPack read at offset {offset}: {read_bytes} != {length}"
            )
        started = time.perf_counter_ns()
        digest = hashlib.sha256(view).hexdigest()
        checksum_ns = time.perf_counter_ns() - started
        expected = self._record_digest(record)
        if digest != expected:
            raise ExpertPackIntegrityError(
                f"ExpertPack payload SHA-256 mismatch at offset {offset}: "
                f"{digest} != {expected}"
            )
        return _ReadResult(read_ns, checksum_ns)

    @staticmethod
    def _host_component(
        staging: torch.Tensor,
        offset: int,
        nbytes: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = staging.narrow(0, offset, nbytes)
        if dtype == torch.float32:
            value = value.view(torch.float32)
        return value.reshape(shape or ())

    def _enqueue_install(
        self, staging: torch.Tensor, slot_index: int
    ) -> torch.cuda.Event:
        assert self._transfer_stream is not None
        runtime = self._slot_runtime[slot_index]
        with torch.cuda.stream(self._transfer_stream):
            if runtime.last_use is not None:
                self._transfer_stream.wait_event(runtime.last_use)
            elif runtime.ready is not None:
                self._transfer_stream.wait_event(runtime.ready)
            for name, offset, nbytes, shape, dtype in _COMPONENT_LAYOUT:
                source = self._host_component(staging, offset, nbytes, shape, dtype)
                self._cache_components[name][slot_index].copy_(
                    source, non_blocking=True
                )
            ready = torch.cuda.Event(enable_timing=False)
            ready.record(self._transfer_stream)
        return ready

    @staticmethod
    def _ordered_unique(values: Iterable[int]) -> tuple[int, ...]:
        return tuple(dict.fromkeys(values))

    def _validate_placements(
        self,
        keys: Sequence[tuple[int, int]],
        placements: Sequence[_Placement],
    ) -> None:
        """Reject corrupt policy output before any payload or slot is touched."""
        if len(placements) != len(keys):
            raise ExpertPackFailedError(
                "expert cache policy returned an incomplete placement mapping"
            )
        slots: set[int] = set()
        for key, placement in zip(keys, placements):
            if placement.key != key:
                raise ExpertPackFailedError(
                    "expert cache policy returned a placement for the wrong key"
                )
            slot = placement.slot
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot < 0
                or slot >= self.cache_capacity
                or slot in slots
            ):
                raise ExpertPackFailedError(
                    f"expert cache policy returned an illegal slot mapping: {slot}"
                )
            slots.add(slot)

    def _validate_assignments(
        self,
        placements: Sequence[_Placement],
        assignments: Sequence[tuple[int, int]],
    ) -> None:
        """Verify the committed mapping before constructing a CUDA-visible map."""
        if len(assignments) != len(placements):
            raise ExpertPackFailedError(
                "expert cache policy committed an incomplete assignment mapping"
            )
        for placement, assignment in zip(placements, assignments):
            if not isinstance(assignment, tuple) or len(assignment) != 2:
                raise ExpertPackFailedError(
                    "expert cache policy returned a malformed assignment"
                )
            slot, epoch = assignment
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot != placement.slot
                or slot < 0
                or slot >= self.cache_capacity
                or isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch < 0
            ):
                raise ExpertPackFailedError(
                    f"expert cache policy returned an illegal assignment: {assignment}"
                )
            current = self._policy.slots[slot]
            if (
                current.key != placement.key
                or current.epoch != epoch
                or current.leases <= 0
                or self._policy.key_to_slot.get(placement.key) != slot
            ):
                raise ExpertPackFailedError(
                    "expert cache assignment does not match committed policy state"
                )

    def acquire(self, layer_id: int, expert_ids: torch.Tensor) -> ExpertLease:
        """Acquire every unique routed expert and publish one layer mapping.

        ``expert_ids`` remains the logical router output.  The returned
        ``expert_to_slot`` maps those global IDs into the physical cache used
        by the slot-aware grouped GEMMs.
        """
        if layer_id not in _ACTIVE_LAYERS:
            raise ValueError("expert offload layer_id must be in 1..38")
        if not isinstance(expert_ids, torch.Tensor):
            raise TypeError("expert_ids must be a torch.Tensor")
        if expert_ids.ndim != 2 or expert_ids.shape[1] != 8 or expert_ids.numel() == 0:
            raise ValueError("expert_ids must be a non-empty [T,8] tensor")
        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("expert_ids must use int32 or int64")
        # Reject a poisoned store before initiating even the route D2H copy.
        with self._lock:
            self._ensure_usable()
        try:
            route_ids = tuple(
                int(value) for value in expert_ids.detach().cpu().reshape(-1).tolist()
            )
        except Exception as error:
            with self._lock:
                self._latch_failure(error, "cuda")
            raise
        if any(expert < 0 or expert >= _NUM_EXPERTS for expert in route_ids):
            raise ValueError("expert_ids contains an out-of-range global expert")
        unique_experts = self._ordered_unique(route_ids)
        keys = tuple((layer_id, expert) for expert in unique_experts)

        with self._lock:
            self._ensure_usable()
            assert self._executor is not None
            assert self._transfer_stream is not None
            assert self._device is not None
            try:
                placements = self._policy.plan(keys)
                self._validate_placements(keys, placements)
            except ExpertCacheBusyError:
                raise
            except Exception as error:
                self._latch_failure(error, "epoch")
                raise
            misses = tuple(placement for placement in placements if not placement.hit)
            ready_by_slot: dict[int, torch.cuda.Event] = {}
            try:
                for start in range(0, len(misses), len(self._staging)):
                    batch = misses[start : start + len(self._staging)]
                    jobs: list[
                        tuple[
                            _Placement,
                            int,
                            torch.Tensor,
                            concurrent.futures.Future[_ReadResult],
                        ]
                    ] = []
                    for placement in batch:
                        stage_index = self._stage_cursor
                        self._stage_cursor = (self._stage_cursor + 1) % len(
                            self._staging
                        )
                        stage_ready = self._staging_ready[stage_index]
                        if stage_ready is not None:
                            waited = time.perf_counter_ns()
                            stage_ready.synchronize()
                            self._stats["stage_wait_ns"] = int(
                                self._stats["stage_wait_ns"]
                            ) + (time.perf_counter_ns() - waited)
                        staging = self._staging[stage_index]
                        record = self.manifest.record(*placement.key)
                        future = self._executor.submit(
                            self._read_payload, record, staging
                        )
                        jobs.append((placement, stage_index, staging, future))

                    for placement, stage_index, staging, future in jobs:
                        result = future.result()
                        self._stats["pack_reads"] = int(self._stats["pack_reads"]) + 1
                        self._stats["pack_read_bytes"] = (
                            int(self._stats["pack_read_bytes"]) + PAYLOAD_SIZE
                        )
                        self._stats["pack_read_ns"] = (
                            int(self._stats["pack_read_ns"]) + result.read_ns
                        )
                        self._stats["checksum_ns"] = (
                            int(self._stats["checksum_ns"]) + result.checksum_ns
                        )
                        try:
                            ready = self._enqueue_install(staging, placement.slot)
                        except Exception as error:
                            raise ExpertPackFailedError(
                                f"CUDA H2D install failed for {placement.key}"
                            ) from error
                        ready_by_slot[placement.slot] = ready
                        self._staging_ready[stage_index] = ready
                        self._stats["h2d_bytes"] = (
                            int(self._stats["h2d_bytes"]) + PAYLOAD_SIZE
                        )
            except ExpertPackIntegrityError as error:
                self._latch_failure(error, "checksum")
                raise
            except OSError as error:
                self._latch_failure(error, "io")
                raise
            except Exception as error:
                self._latch_failure(error, "cuda")
                raise

            evictions = sum(
                not placement.hit and placement.prior_key is not None
                for placement in placements
            )
            try:
                assignments = self._policy.commit_and_lease(placements)
                self._validate_assignments(placements, assignments)
            except Exception as error:
                self._latch_failure(error, "epoch")
                raise

            try:
                for placement in misses:
                    runtime = self._slot_runtime[placement.slot]
                    runtime.ready = ready_by_slot[placement.slot]
                    runtime.last_use = None

                current_stream = torch.cuda.current_stream(self._device)
                for placement in placements:
                    ready = self._slot_runtime[placement.slot].ready
                    if ready is not None:
                        current_stream.wait_event(ready)
                mapping_cpu = torch.full((_NUM_EXPERTS,), -1, dtype=torch.int32)
                for expert, (slot, _) in zip(unique_experts, assignments):
                    mapping_cpu[expert] = slot
                expert_to_slot = mapping_cpu.to(self._device, non_blocking=False)
            except Exception as error:
                self._latch_failure(error, "cuda")
                raise

            self._stats["cache_hits"] = int(self._stats["cache_hits"]) + (
                len(placements) - len(misses)
            )
            self._stats["cache_misses"] = int(self._stats["cache_misses"]) + len(misses)
            self._stats["cache_evictions"] = (
                int(self._stats["cache_evictions"]) + evictions
            )
            self._route_calls_by_layer[layer_id] += 1
            self._route_tokens_by_layer[layer_id] += int(expert_ids.shape[0])
            self._unique_experts_peak_by_layer[layer_id] = max(
                self._unique_experts_peak_by_layer[layer_id], len(unique_experts)
            )
            return ExpertLease(
                self,
                layer_id=layer_id,
                expert_ids=unique_experts,
                assignments=assignments,
                gate_up=self.gate_up,
                down=self.down,
                expert_to_slot=expert_to_slot,
            )

    def _release(
        self,
        assignments: Sequence[tuple[int, int]],
        stream: torch.cuda.Stream | None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            if self._device is None:
                raise ExpertPackStoreError("expert device cache is not initialized")
            if stream is None:
                stream = torch.cuda.current_stream(self._device)
            try:
                event = torch.cuda.Event(enable_timing=False)
                event.record(stream)
            except Exception as error:
                self._latch_failure(error, "cuda")
                raise
            try:
                self._policy.release(assignments)
            except Exception as error:
                self._latch_failure(error, "epoch")
                raise
            for slot_index, _ in assignments:
                self._slot_runtime[slot_index].last_use = event

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            value: dict[str, Any] = dict(self._stats)
            value.update(
                {
                    "state": self._state,
                    "failure": repr(self._failure) if self._failure else None,
                    "manifest_path": str(self.config.manifest_path),
                    "pack_path": str(self.pack_path),
                    "resident_experts": self._policy.resident,
                    "resident_bytes": self._policy.resident * PAYLOAD_SIZE,
                    "route_calls_by_layer": dict(self._route_calls_by_layer),
                    "route_tokens_by_layer": dict(self._route_tokens_by_layer),
                    "unique_experts_peak_by_layer": dict(
                        self._unique_experts_peak_by_layer
                    ),
                }
            )
            reads = int(value["pack_reads"])
            value["mean_read_ms"] = (
                int(value["pack_read_ns"]) / reads / 1e6 if reads else 0.0
            )
            value["mean_checksum_ms"] = (
                int(value["checksum_ns"]) / reads / 1e6 if reads else 0.0
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
            executor, self._executor = self._executor, None
            try:
                if self._device is not None:
                    torch.cuda.synchronize(self._device)
            except Exception as error:
                if self._failure is None:
                    self._failure = error
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            try:
                self._write_stats()
            except Exception:
                logger.exception("failed to write Qwen3.5 ExpertPack statistics")
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            self._state = "CLOSED"
            self._cache_components.clear()
            self._gate_up = None
            self._down = None
            self._staging.clear()
            self._staging_ready.clear()
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
]
