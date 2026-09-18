# SPDX-License-Identifier: Apache-2.0
"""CPU-source storage and bounded GPU-cache primitives for routed MoE experts."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import torch


class StreamLike(Protocol):
    def wait_event(self, event) -> None: ...


class EventLike(Protocol):
    def record(self, stream=None) -> None: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class HostExpertLayer:
    """Strongly owned, pageable final-layout expert tensors for one MoE layer."""

    layer_id: int
    top_k: int
    num_experts: int
    tensors: Mapping[str, torch.Tensor]
    expert_tensor_bytes: int
    staging_payload_bytes: int
    total_bytes: int
    staging_layout: Mapping[str, tuple[int, int]]

    @classmethod
    def from_tensors(
        cls,
        *,
        layer_id: int,
        top_k: int,
        num_experts: int,
        tensors: Mapping[str, torch.Tensor],
    ) -> HostExpertLayer:
        if layer_id < 0 or top_k <= 0 or num_experts <= 0:
            raise ValueError("layer_id, top_k, and num_experts must be positive")
        if not tensors:
            raise ValueError("a host expert layer needs at least one tensor")
        payload_bytes = 0
        tensor_bytes = 0
        retained: dict[str, torch.Tensor] = {}
        staging_layout: dict[str, tuple[int, int]] = {}
        for name, tensor in tensors.items():
            if tensor.device.type != "cpu" or tensor.is_pinned():
                raise ValueError(f"source {name!r} must be pageable CPU memory")
            if tensor.ndim == 0 or tensor.shape[0] != num_experts:
                raise ValueError(f"source {name!r} is not expert-major")
            if not tensor.is_contiguous():
                raise ValueError(f"source {name!r} must be contiguous")
            retained[name] = tensor
            byte_count = tensor[0].numel() * tensor.element_size()
            tensor_bytes += byte_count
            alignment = max(16, tensor.element_size())
            payload_bytes = _align_up(payload_bytes, alignment)
            staging_layout[name] = (payload_bytes, byte_count)
            payload_bytes += byte_count
        payload_bytes = _align_up(payload_bytes, 16)
        return cls(
            layer_id=layer_id,
            top_k=top_k,
            num_experts=num_experts,
            tensors=retained,
            expert_tensor_bytes=tensor_bytes,
            staging_payload_bytes=payload_bytes,
            total_bytes=tensor_bytes * num_experts,
            staging_layout=staging_layout,
        )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class _StagingSlot:
    buffer: torch.Tensor
    h2d_done: EventLike | None = None


@dataclass
class _PendingTiming:
    """A completed-on-stream timing interval that has not completed on GPU."""

    metric: str
    start: object
    end: object


class PinnedStagingRing:
    """Fixed-size pageable-to-pinned staging ring shared by all cache pools.

    The ring is the only host memory pinned by this backend.  Before a slot is
    overwritten on CPU, its previous H2D event is synchronized: asynchronous
    DMA may still read the pinned bytes even after enqueueing completed.
    """

    def __init__(
        self,
        stage_slots: int,
        max_expert_payload_bytes: int,
        *,
        event_factory: Callable[[], EventLike],
        buffer_factory: Callable[[int], torch.Tensor] | None = None,
        require_pinned: bool = True,
        staging_wait_callback: Callable[[int], None] | None = None,
    ) -> None:
        if stage_slots <= 0 or max_expert_payload_bytes <= 0:
            raise ValueError(
                "stage_slots and max_expert_payload_bytes must be positive"
            )
        if buffer_factory is None:
            buffer_factory = lambda size: torch.empty(
                size, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        self.max_expert_payload_bytes = max_expert_payload_bytes
        self._event_factory = event_factory
        self._slots = [
            _StagingSlot(buffer_factory(max_expert_payload_bytes))
            for _ in range(stage_slots)
        ]
        if require_pinned and any(not slot.buffer.is_pinned() for slot in self._slots):
            raise RuntimeError("PinnedStagingRing requires pinned CPU buffers")
        self._next_slot = 0
        self._closed = False
        self._staging_wait_callback = staging_wait_callback

    @property
    def allocated_bytes(self) -> int:
        return len(self._slots) * self.max_expert_payload_bytes

    def stage(
        self, host_layer: HostExpertLayer, expert_id: int
    ) -> tuple[_StagingSlot, dict[str, torch.Tensor]]:
        if self._closed:
            raise RuntimeError("staging ring is closed")
        if not 0 <= expert_id < host_layer.num_experts:
            raise IndexError(
                f"expert {expert_id} is outside layer {host_layer.layer_id}"
            )
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        if slot.h2d_done is not None:
            # This is a host-side wait, not a stream wait: CPU is about to
            # overwrite the source buffer of the prior nonblocking H2D.
            wait_start_ns = time.perf_counter_ns()
            slot.h2d_done.synchronize()
            if self._staging_wait_callback is not None:
                self._staging_wait_callback(time.perf_counter_ns() - wait_start_ns)
            slot.h2d_done = None

        views: dict[str, torch.Tensor] = {}
        for name, source in host_layer.tensors.items():
            expert = source[expert_id]
            offset, byte_count = host_layer.staging_layout[name]
            end = offset + byte_count
            if end > self.max_expert_payload_bytes:
                raise RuntimeError("staging payload exceeds ring slot capacity")
            byte_view = slot.buffer.narrow(0, offset, byte_count)
            byte_view.copy_(expert.view(torch.uint8).reshape(-1))
            views[name] = byte_view.view(expert.dtype).reshape(expert.shape)
        return slot, views

    def mark_h2d_submitted(self, slot: _StagingSlot, transfer_stream) -> None:
        event = self._event_factory()
        event.record(transfer_stream)
        slot.h2d_done = event

    def close(self) -> None:
        self.synchronize_pending()
        self._finalize_close()

    def _finalize_close(self) -> None:
        self._closed = True
        self._slots.clear()

    def synchronize_pending(self) -> None:
        for slot in self._slots:
            if slot.h2d_done is not None:
                slot.h2d_done.synchronize()
                slot.h2d_done = None


@dataclass
class _CacheSlot:
    logical_expert_id: int | None = None
    generation: int = 0
    active_leases: int = 0
    frequency: int = 0
    last_access: int = 0
    ready: EventLike | None = None
    last_use: EventLike | None = None


@dataclass(frozen=True)
class ExpertCacheLease:
    """A generation-bound lease returned by ``ExpertCachePool.acquire``."""

    pool: ExpertCachePool
    bindings: Mapping[int, tuple[int, int]]
    token: int

    @property
    def logical_to_physical(self) -> dict[int, int]:
        return {logical: slot for logical, (slot, _) in self.bindings.items()}

    def release_after(self, compute_stream) -> None:
        self.pool.release_after(self, compute_stream)


@contextmanager
def _stream_scope(stream):
    """Use CUDA stream context when available, otherwise keep fake tests CPU-only."""
    if stream is None:
        yield
    elif hasattr(stream, "__enter__"):
        with stream:
            yield
    elif torch.cuda.is_available() and isinstance(stream, torch.cuda.Stream):
        with torch.cuda.stream(stream):
            yield
    else:
        with nullcontext():
            yield


class ExpertCachePool:
    """Independent per-layer cache with frequency+LRU eviction and leases."""

    def __init__(
        self,
        host_layer: HostExpertLayer,
        cache_slots: int,
        staging_ring: PinnedStagingRing,
        *,
        event_factory: Callable[[], EventLike],
        tensor_factory: (
            Callable[[tuple[int, ...], torch.dtype], torch.Tensor] | None
        ) = None,
        route_complete_callback: Callable[[], None] | None = None,
        timing_event_factory: Callable[[], object] | None = None,
    ) -> None:
        if cache_slots < host_layer.top_k:
            raise ValueError("each cache pool needs at least top_k slots")
        if cache_slots > host_layer.num_experts:
            cache_slots = host_layer.num_experts
        if tensor_factory is None:
            tensor_factory = lambda shape, dtype: torch.empty(
                shape, dtype=dtype, device="cuda"
            )
        self.host_layer = host_layer
        self.staging_ring = staging_ring
        self._event_factory = event_factory
        self.cache_tensors = {
            name: tensor_factory((cache_slots, *tensor.shape[1:]), tensor.dtype)
            for name, tensor in host_layer.tensors.items()
        }
        self._slots = [_CacheSlot() for _ in range(cache_slots)]
        self._clock = 0
        self._leases: dict[int, ExpertCacheLease] = {}
        self._next_lease_token = 0
        self._closed = False
        self._poisoned: BaseException | None = None
        self._route_complete_callback = route_complete_callback
        # Timing events are deliberately distinct from ready/last-use events.
        # Readiness and lifetime events cannot enable timing because they are
        # correctness fences with different ownership and synchronization.
        self._timing_event_factory = timing_event_factory
        self._pending_timing: list[_PendingTiming] = []
        self.stats = {
            "acquire_calls": 0,
            "requested_experts": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "h2d_bytes": 0,
            "h2d_time_ns": 0,
            "ready_wait_count": 0,
            # Filled only by distinct CUDA timing events around the compute
            # stream dependency; remains zero for CPU-only/fake event tests.
            "ready_wait_time_ns": 0,
        }

    @property
    def cache_slots(self) -> int:
        return len(self._slots)

    @property
    def allocated_bytes(self) -> int:
        return self.cache_slots * self.host_layer.expert_tensor_bytes

    def _next_clock(self) -> int:
        self._clock += 1
        return self._clock

    def _find_slot(self, logical_expert_id: int) -> int | None:
        for index, slot in enumerate(self._slots):
            if slot.logical_expert_id == logical_expert_id:
                return index
        return None

    def _select_victim(self) -> int:
        empty = next(
            (
                index
                for index, slot in enumerate(self._slots)
                if slot.logical_expert_id is None
            ),
            None,
        )
        if empty is not None:
            return empty
        candidates = [
            (slot.frequency, slot.last_access, index)
            for index, slot in enumerate(self._slots)
            if slot.active_leases == 0
        ]
        if not candidates:
            raise RuntimeError("all expert cache slots have active leases")
        return min(candidates)[2]

    def _load_expert(
        self, slot_index: int, logical_expert_id: int, transfer_stream
    ) -> None:
        slot = self._slots[slot_index]
        try:
            if slot.last_use is not None:
                transfer_stream.wait_event(slot.last_use)
            if slot.logical_expert_id is not None:
                self.stats["evictions"] += 1

            staging_slot, staged_tensors = self.staging_ring.stage(
                self.host_layer, logical_expert_id
            )
            with _stream_scope(transfer_stream):
                timing_start = (
                    self._timing_event_factory()
                    if self._timing_event_factory is not None
                    else None
                )
                if timing_start is not None:
                    timing_start.record(transfer_stream)
                for name, staged in staged_tensors.items():
                    self.cache_tensors[name][slot_index].copy_(
                        staged, non_blocking=True
                    )
                ready = self._event_factory()
                ready.record(transfer_stream)
                # The H2D event is separate from ``ready`` because its purpose
                # is host-buffer reuse; ready gates cache-data consumption.
                self.staging_ring.mark_h2d_submitted(staging_slot, transfer_stream)
                if timing_start is not None:
                    timing_end = self._timing_event_factory()
                    timing_end.record(transfer_stream)
                    self._pending_timing.append(
                        _PendingTiming("h2d", timing_start, timing_end)
                    )
        except BaseException as error:
            # A failed copy might have partially overwritten the storage.  Do
            # not leave its old logical mapping eligible for a cache hit, and
            # poison the pool because an in-flight transfer cannot be safely
            # ordered for reuse after an arbitrary enqueue failure.
            slot.logical_expert_id = None
            slot.generation += 1
            slot.ready = None
            slot.last_use = None
            self._poisoned = error
            raise RuntimeError("expert cache pool was poisoned during H2D") from error

        slot.logical_expert_id = logical_expert_id
        slot.generation += 1
        slot.frequency = 0
        slot.ready = ready
        slot.last_use = None
        self.stats["h2d_bytes"] += self.host_layer.expert_tensor_bytes

    def drain_timing(self, *, force: bool = False) -> None:
        """Collect only CUDA intervals whose end event has completed.

        Statistics must never make the hot path synchronize.  CUDA timing is
        therefore drained with ``Event.query`` during snapshots; shutdown is
        the sole forced path.  Test doubles without timing APIs are ignored so
        the cache's correctness event abstraction stays intentionally small.
        """
        pending: list[_PendingTiming] = []
        for interval in self._pending_timing:
            query = getattr(interval.end, "query", None)
            elapsed_time = getattr(interval.start, "elapsed_time", None)
            if not callable(query) or not callable(elapsed_time):
                continue
            if force:
                interval.end.synchronize()
            elif not query():
                pending.append(interval)
                continue
            try:
                # torch reports CUDA event elapsed time in milliseconds.
                elapsed_ns = int(elapsed_time(interval.end) * 1_000_000)
            except (RuntimeError, TypeError):
                # A timing implementation can reject an event pair after a
                # device error.  Keep data-plane safety independent of stats.
                continue
            self.stats[f"{interval.metric}_time_ns"] += max(0, elapsed_ns)
        self._pending_timing = pending

    def acquire(
        self, logical_expert_ids, transfer_stream, compute_stream
    ) -> ExpertCacheLease:
        if self._closed:
            raise RuntimeError("expert cache pool is closed")
        if self._poisoned is not None:
            raise RuntimeError("expert cache pool is poisoned") from self._poisoned
        unique_ids = list(
            dict.fromkeys(int(expert_id) for expert_id in logical_expert_ids)
        )
        if not unique_ids:
            raise ValueError("expert cache acquisition requires at least one expert")
        if len(unique_ids) > self.cache_slots:
            raise RuntimeError("requested expert union exceeds cache capacity")
        if any(
            expert_id < 0 or expert_id >= self.host_layer.num_experts
            for expert_id in unique_ids
        ):
            raise IndexError("expert cache request contains an invalid logical expert")

        self.stats["acquire_calls"] += 1
        self.stats["requested_experts"] += len(unique_ids)
        bindings: dict[int, tuple[int, int]] = {}
        acquired_slots: list[_CacheSlot] = []
        try:
            for logical_expert_id in unique_ids:
                slot_index = self._find_slot(logical_expert_id)
                if slot_index is None:
                    self.stats["cache_misses"] += 1
                    slot_index = self._select_victim()
                    self._load_expert(slot_index, logical_expert_id, transfer_stream)
                else:
                    self.stats["cache_hits"] += 1
                slot = self._slots[slot_index]
                slot.active_leases += 1
                acquired_slots.append(slot)
                slot.frequency += 1
                slot.last_access = self._next_clock()
                if slot.ready is not None:
                    # Keep exactly one dependency for each acquired ready
                    # slot.  These timing events are separate from ``ready``:
                    # they measure the actual compute-stream wait without
                    # changing the correctness fence's lifetime semantics.
                    timing_start = (
                        self._timing_event_factory()
                        if self._timing_event_factory is not None
                        else None
                    )
                    if timing_start is not None:
                        timing_start.record(compute_stream)
                    compute_stream.wait_event(slot.ready)
                    self.stats["ready_wait_count"] += 1
                    if timing_start is not None:
                        timing_end = self._timing_event_factory()
                        timing_end.record(compute_stream)
                        self._pending_timing.append(
                            _PendingTiming("ready_wait", timing_start, timing_end)
                        )
                bindings[logical_expert_id] = (slot_index, slot.generation)
        except BaseException:
            for slot in acquired_slots:
                slot.active_leases -= 1
            raise

        self._next_lease_token += 1
        lease = ExpertCacheLease(self, bindings, self._next_lease_token)
        self._leases[lease.token] = lease
        return lease

    def release_after(self, lease: ExpertCacheLease, compute_stream) -> None:
        if lease.pool is not self or self._leases.get(lease.token) is not lease:
            raise RuntimeError("stale or double expert cache lease release")
        validated: list[_CacheSlot] = []
        for logical_expert_id, (slot_index, generation) in lease.bindings.items():
            slot = self._slots[slot_index]
            if (
                slot.logical_expert_id != logical_expert_id
                or slot.generation != generation
                or slot.active_leases <= 0
            ):
                raise RuntimeError("stale expert cache lease generation")
            validated.append(slot)
        last_use_events = [self._event_factory() for _ in validated]
        for event in last_use_events:
            event.record(compute_stream)
        for slot, last_use in zip(validated, last_use_events):
            # Record after the consumer kernel has been enqueued.  A transfer
            # stream waits this event before overwriting a future victim slot.
            slot.last_use = last_use
            slot.active_leases -= 1
        del self._leases[lease.token]
        if self._route_complete_callback is not None:
            self._route_complete_callback()

    def close(self) -> None:
        if self._leases:
            raise RuntimeError("cannot close expert cache pool with active leases")
        self.synchronize_pending()
        self.drain_timing(force=True)
        self._finalize_close()

    def synchronize_pending(self) -> None:
        if self._leases:
            raise RuntimeError(
                "cannot synchronize expert cache pool with active leases"
            )
        for slot in self._slots:
            if slot.ready is not None:
                slot.ready.synchronize()
            if slot.last_use is not None:
                slot.last_use.synchronize()

    def _finalize_close(self) -> None:
        self._closed = True
        self.cache_tensors.clear()


class CpuMemoryExpertBackend:
    """Own host expert sources, shared staging, independent per-layer pools."""

    def __init__(self) -> None:
        self.host_layers: dict[int, HostExpertLayer] = {}
        self.pools: dict[int, ExpertCachePool] = {}
        self.staging_ring: PinnedStagingRing | None = None
        self._stats_path: str | Path | None = None
        self._stats_flush_interval = 0
        self._completed_routes = 0
        self._stats = {
            "moe_calls": 0,
            "microbatches": 0,
            "staging_wait_count": 0,
            "staging_wait_ns": 0,
        }

    def configure_stats(self, path: str | Path | None, interval: int) -> None:
        if interval < 0:
            raise ValueError("stats_flush_interval must be non-negative")
        self._stats_path = path
        self._stats_flush_interval = interval

    def _record_completed_route(self) -> None:
        self._completed_routes += 1
        if (
            self._stats_path is not None
            and self._stats_flush_interval > 0
            and self._completed_routes % self._stats_flush_interval == 0
        ):
            self.flush_stats(self._stats_path)

    def record_moe_call(self) -> None:
        self._stats["moe_calls"] += 1

    def record_microbatch(self) -> None:
        self._stats["microbatches"] += 1

    def _record_staging_wait(self, elapsed_ns: int) -> None:
        self._stats["staging_wait_count"] += 1
        self._stats["staging_wait_ns"] += elapsed_ns

    def register_host_layer(
        self,
        *,
        layer_id: int,
        top_k: int,
        num_experts: int,
        tensors: Mapping[str, torch.Tensor],
    ) -> HostExpertLayer:
        if self.pools:
            raise RuntimeError("cannot replace host layers after cache configuration")
        host_layer = HostExpertLayer.from_tensors(
            layer_id=layer_id,
            top_k=top_k,
            num_experts=num_experts,
            tensors=tensors,
        )
        if layer_id in self.host_layers:
            raise RuntimeError(
                f"duplicate expert source registration for layer {layer_id}"
            )
        self.host_layers[layer_id] = host_layer
        return host_layer

    def initialize_cuda(
        self,
        *,
        target_device: torch.device | str,
        cache_vram_mib: int,
        cache_vram_reserve_mib: int,
        stage_slots: int,
    ) -> Mapping[int, ExpertCachePool]:
        """Create CUDA cache storage and the dedicated pageable-source stream."""
        device = torch.device(target_device)
        if device.type != "cuda":
            raise ValueError("CPU-memory expert cache requires a CUDA target device")
        self.target_device = device
        self.transfer_stream = torch.cuda.Stream(device=device)
        return self.configure_cache(
            cache_vram_mib=cache_vram_mib,
            cache_vram_reserve_mib=cache_vram_reserve_mib,
            stage_slots=stage_slots,
            event_factory=lambda: torch.cuda.Event(),
            timing_event_factory=lambda: torch.cuda.Event(enable_timing=True),
            tensor_factory=lambda shape, dtype: torch.empty(
                shape, dtype=dtype, device=device
            ),
        )

    def configure_cache(
        self,
        *,
        cache_vram_mib: int,
        cache_vram_reserve_mib: int,
        stage_slots: int,
        event_factory: Callable[[], EventLike],
        tensor_factory: (
            Callable[[tuple[int, ...], torch.dtype], torch.Tensor] | None
        ) = None,
        staging_buffer_factory: Callable[[int], torch.Tensor] | None = None,
        require_pinned_staging: bool = True,
        free_bytes_provider: Callable[[], int | tuple[int, int]] | None = None,
        timing_event_factory: Callable[[], object] | None = None,
    ) -> Mapping[int, ExpertCachePool]:
        if self.pools or self.staging_ring is not None:
            raise RuntimeError("expert cache backend is already configured")
        if cache_vram_mib <= 0 or cache_vram_reserve_mib < 0:
            raise ValueError("invalid cache_vram_mib or cache_vram_reserve_mib")
        if not self.host_layers:
            raise RuntimeError("no host expert layers are registered")
        if free_bytes_provider is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA free-memory query requires an injected provider"
                )
            free_bytes_provider = torch.cuda.mem_get_info
        free_result = free_bytes_provider()
        free_bytes = free_result[0] if isinstance(free_result, tuple) else free_result
        requested_bytes = cache_vram_mib * 1024 * 1024
        reserve_bytes = cache_vram_reserve_mib * 1024 * 1024
        budget = min(requested_bytes, max(0, free_bytes - reserve_bytes))
        self.requested_cache_bytes = requested_bytes
        self.free_cache_bytes = free_bytes
        self.cache_budget_bytes = budget
        layers = list(self.host_layers.values())
        slots = {
            layer.layer_id: min(layer.top_k, layer.num_experts) for layer in layers
        }
        minimum_bytes = sum(
            slots[layer.layer_id] * layer.expert_tensor_bytes for layer in layers
        )
        if minimum_bytes > budget:
            raise ValueError("cache budget cannot provide top_k slots for every layer")
        remaining = budget - minimum_bytes
        # Allocate remaining whole slots to the least-covered layer first.
        # This normalizes capacity by expert count and avoids starving layers
        # with a large expert table merely because their payload is smaller.
        while True:
            candidates = [
                layer
                for layer in layers
                if slots[layer.layer_id] < layer.num_experts
                and layer.expert_tensor_bytes <= remaining
            ]
            if not candidates:
                break
            layer = min(
                candidates,
                key=lambda item: (
                    slots[item.layer_id] / item.num_experts,
                    item.layer_id,
                ),
            )
            slots[layer.layer_id] += 1
            remaining -= layer.expert_tensor_bytes

        ring = None
        pools: dict[int, ExpertCachePool] = {}
        try:
            ring = PinnedStagingRing(
                stage_slots,
                max(layer.staging_payload_bytes for layer in layers),
                event_factory=event_factory,
                buffer_factory=staging_buffer_factory,
                require_pinned=require_pinned_staging,
                staging_wait_callback=self._record_staging_wait,
            )
            for layer in layers:
                pools[layer.layer_id] = ExpertCachePool(
                    layer,
                    slots[layer.layer_id],
                    ring,
                    event_factory=event_factory,
                    tensor_factory=tensor_factory,
                    route_complete_callback=self._record_completed_route,
                    timing_event_factory=timing_event_factory,
                )
        except BaseException:
            for pool in pools.values():
                pool.close()
            if ring is not None:
                ring.close()
            # Keep registration intact and configuration empty so callers can
            # retry after a transient allocation failure.
            self.staging_ring = None
            self.pools = {}
            raise
        self.staging_ring = ring
        self.pools = pools
        return self.pools

    def snapshot_stats(self) -> dict:
        """Build a nonblocking accounting snapshot suitable for JSON output."""
        for pool in self.pools.values():
            pool.drain_timing()
        allocated_cache_bytes = sum(
            pool.allocated_bytes for pool in self.pools.values()
        )
        peak_gpu_memory_bytes = 0
        device = getattr(self, "target_device", None)
        if device is not None and torch.cuda.is_available():
            try:
                peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
            except (RuntimeError, TypeError):
                # Device teardown or a mocked CUDA environment must not make
                # a diagnostic flush fail after useful work completed.
                pass
        return {
            "totals": {
                "host_expert_bytes": sum(
                    layer.total_bytes for layer in self.host_layers.values()
                ),
                "cache_allocated_bytes": allocated_cache_bytes,
                "staging_allocated_bytes": (
                    self.staging_ring.allocated_bytes
                    if self.staging_ring is not None
                    else 0
                ),
                "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                **self._stats,
            },
            "host_layers": {
                str(layer_id): {
                    "top_k": layer.top_k,
                    "num_experts": layer.num_experts,
                    "expert_tensor_bytes": layer.expert_tensor_bytes,
                    "staging_payload_bytes": layer.staging_payload_bytes,
                    "total_bytes": layer.total_bytes,
                }
                for layer_id, layer in self.host_layers.items()
            },
            "pools": {
                str(layer_id): {
                    "cache_slots": pool.cache_slots,
                    # Concise aliases make the per-layer schema easier for
                    # dashboards while preserving older counter names.
                    "hits": pool.stats["cache_hits"],
                    "misses": pool.stats["cache_misses"],
                    **pool.stats,
                }
                for layer_id, pool in self.pools.items()
            },
            "cache_budget": {
                "requested_bytes": getattr(self, "requested_cache_bytes", 0),
                "free_bytes": getattr(self, "free_cache_bytes", 0),
                "budget_bytes": getattr(self, "cache_budget_bytes", 0),
                # This is real tensor allocation, not the possibly-unused
                # budget remaining after whole-slot allocation.
                "actual_bytes": allocated_cache_bytes,
            },
        }

    def flush_stats(self, path: str | Path) -> None:
        payload = self.snapshot_stats()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "w") as file:
                file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def close(self) -> None:
        if any(pool._leases for pool in self.pools.values()):
            raise RuntimeError("cannot close expert backend with active leases")
        # Synchronize every cache and staging event before releasing any CUDA
        # storage.  This prevents a caching allocator from reusing bytes that
        # an asynchronous transfer or compute stream still references.
        for pool in self.pools.values():
            pool.synchronize_pending()
            pool.drain_timing(force=True)
        if self.staging_ring is not None:
            self.staging_ring.synchronize_pending()
        for pool in self.pools.values():
            pool._finalize_close()
        if self.staging_ring is not None:
            self.staging_ring._finalize_close()
