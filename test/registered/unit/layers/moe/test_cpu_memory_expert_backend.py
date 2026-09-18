import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.cpu_memory_expert_backend import (
    CpuMemoryExpertBackend,
    ExpertCachePool,
    HostExpertLayer,
    PinnedStagingRing,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-b-test-cpu-intel")


class _FakeEvent:
    def __init__(self):
        self.recorded_on = []
        self.synchronize_calls = 0

    def record(self, stream=None):
        self.recorded_on.append(stream)

    def synchronize(self):
        self.synchronize_calls += 1


class _TimingEvent(_FakeEvent):
    """Timing-capable fake kept separate from correctness fence events."""

    def query(self):
        return True

    def elapsed_time(self, other):
        return 2.5


class _FakeStream:
    def __init__(self):
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


class _Events:
    def __init__(self):
        self.events = []

    def make(self):
        event = _FakeEvent()
        self.events.append(event)
        return event


def _cpu_buffer(size):
    return torch.empty(size, dtype=torch.uint8)


def _cpu_cache(shape, dtype):
    return torch.empty(shape, dtype=dtype)


def _host_layer(layer_id=0, num_experts=3, top_k=1):
    weights = torch.arange(num_experts * 2, dtype=torch.float32).reshape(num_experts, 2)
    scales = torch.arange(num_experts, dtype=torch.float32).reshape(num_experts, 1)
    return HostExpertLayer.from_tensors(
        layer_id=layer_id,
        top_k=top_k,
        num_experts=num_experts,
        tensors={"weights": weights, "scales": scales},
    )


class TestCpuMemoryExpertBackend(unittest.TestCase):
    def test_staging_ring_waits_before_cpu_buffer_overwrite(self):
        events = _Events()
        ring = PinnedStagingRing(
            1,
            _host_layer().staging_payload_bytes,
            event_factory=events.make,
            buffer_factory=_cpu_buffer,
            require_pinned=False,
        )
        host = _host_layer()
        self.assertEqual(host.expert_tensor_bytes, 12)
        self.assertEqual(host.staging_payload_bytes, 32)
        slot, staged = ring.stage(host, 0)
        torch.testing.assert_close(staged["weights"], host.tensors["weights"][0])
        stream = _FakeStream()
        ring.mark_h2d_submitted(slot, stream)
        previous_event = slot.h2d_done

        _, staged = ring.stage(host, 1)

        assert previous_event is not None
        self.assertEqual(previous_event.synchronize_calls, 1)
        torch.testing.assert_close(staged["weights"], host.tensors["weights"][1])

    def test_zero_byte_source_does_not_force_staging(self):
        host = HostExpertLayer.from_tensors(
            layer_id=0,
            top_k=1,
            num_experts=2,
            tensors={"empty": torch.empty(2, 0)},
        )
        ring = PinnedStagingRing(
            1,
            max(1, host.staging_payload_bytes),
            event_factory=_FakeEvent,
            buffer_factory=_cpu_buffer,
            require_pinned=False,
        )

        slot, staged = ring.stage(host, 0)

        self.assertIsNone(slot)
        self.assertEqual(staged["empty"].numel(), 0)

    def test_leases_deduplicate_wait_and_protect_active_victims(self):
        events = _Events()
        host = _host_layer()
        ring = PinnedStagingRing(
            1,
            host.staging_payload_bytes,
            event_factory=events.make,
            buffer_factory=_cpu_buffer,
            require_pinned=False,
        )
        pool = ExpertCachePool(
            host, 1, ring, event_factory=events.make, tensor_factory=_cpu_cache
        )
        transfer = _FakeStream()
        compute = _FakeStream()

        lease = pool.acquire([0, 0], transfer, compute)
        self.assertEqual(lease.logical_to_physical, {0: 0})
        self.assertEqual(pool.stats["cache_misses"], 1)
        self.assertEqual(pool.stats["requested_experts"], 1)
        self.assertEqual(len(compute.waited_events), 1)
        self.assertEqual(pool.stats["ready_wait_count"], 1)
        self.assertEqual(pool.stats["ready_wait_time_ns"], 0)
        with self.assertRaisesRegex(RuntimeError, "active leases"):
            pool.acquire([1], transfer, compute)
        with self.assertRaisesRegex(RuntimeError, "active leases"):
            pool.close()

        lease.release_after(compute)
        with self.assertRaisesRegex(RuntimeError, "stale or double"):
            lease.release_after(compute)
        replacement = pool.acquire([1], transfer, compute)
        self.assertEqual(replacement.logical_to_physical, {1: 0})
        self.assertGreaterEqual(pool.stats["evictions"], 1)
        self.assertTrue(transfer.waited_events)
        replacement.release_after(compute)
        pending_last_use = events.events[-1]
        pool.close()
        self.assertGreaterEqual(pending_last_use.synchronize_calls, 1)

    def test_failed_h2d_invalidates_mapping_and_poisons_pool(self):
        events = _Events()
        host = _host_layer()
        ring = PinnedStagingRing(
            1,
            host.staging_payload_bytes,
            event_factory=events.make,
            buffer_factory=_cpu_buffer,
            require_pinned=False,
        )
        pool = ExpertCachePool(
            host, 1, ring, event_factory=events.make, tensor_factory=_cpu_cache
        )

        class _FailingTensor:
            def __getitem__(self, index):
                return self

            def copy_(self, source, non_blocking):
                raise RuntimeError("injected H2D failure")

        pool.cache_tensors["weights"] = _FailingTensor()
        transfer = _FakeStream()
        compute = _FakeStream()
        with self.assertRaisesRegex(RuntimeError, "poisoned during H2D"):
            pool.acquire([0], transfer, compute)
        self.assertIsNone(pool._slots[0].logical_expert_id)
        with self.assertRaisesRegex(RuntimeError, "pool is poisoned"):
            pool.acquire([1], transfer, compute)

    def test_multi_expert_acquire_rolls_back_prior_lease_counts(self):
        events = _Events()
        host = _host_layer()
        ring = PinnedStagingRing(
            1,
            host.staging_payload_bytes,
            event_factory=events.make,
            buffer_factory=_cpu_buffer,
            require_pinned=False,
        )
        pool = ExpertCachePool(
            host, 2, ring, event_factory=events.make, tensor_factory=_cpu_cache
        )

        class _FailOnSecondWait(_FakeStream):
            def wait_event(self, event):
                super().wait_event(event)
                if len(self.waited_events) == 2:
                    raise RuntimeError("second expert wait failed")

        with self.assertRaisesRegex(RuntimeError, "second expert"):
            pool.acquire([0, 1], _FakeStream(), _FailOnSecondWait())
        self.assertEqual([slot.active_leases for slot in pool._slots], [0, 0])

    def test_configuration_failure_releases_partial_state_for_retry(self):
        events = _Events()
        backend = CpuMemoryExpertBackend()
        backend.register_host_layer(
            layer_id=0,
            top_k=1,
            num_experts=2,
            tensors={"w": torch.empty(2, 262144)},
        )
        backend.register_host_layer(
            layer_id=1,
            top_k=1,
            num_experts=2,
            tensors={"w": torch.empty(2, 262144)},
        )
        calls = 0

        def fail_first_cache(shape, dtype):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected allocation failure")
            return _cpu_cache(shape, dtype)

        kwargs = dict(
            cache_vram_mib=4,
            cache_vram_reserve_mib=0,
            stage_slots=1,
            event_factory=events.make,
            staging_buffer_factory=_cpu_buffer,
            require_pinned_staging=False,
            free_bytes_provider=lambda: 4 * 1024 * 1024,
        )
        with self.assertRaisesRegex(RuntimeError, "injected allocation"):
            backend.configure_cache(tensor_factory=fail_first_cache, **kwargs)
        self.assertEqual(backend.pools, {})
        self.assertIsNone(backend.staging_ring)
        backend.configure_cache(tensor_factory=_cpu_cache, **kwargs)
        backend.close()

    def test_budget_assignment_balances_normalized_slot_coverage(self):
        events = _Events()
        backend = CpuMemoryExpertBackend()
        # One expert payload is exactly one MiB.  A four-MiB budget gives both
        # four-expert layers their top-k slot plus one balanced extra slot.
        shape = (4, 262144)
        backend.register_host_layer(
            layer_id=0,
            top_k=1,
            num_experts=4,
            tensors={"w": torch.empty(shape, dtype=torch.float32)},
        )
        backend.register_host_layer(
            layer_id=1,
            top_k=1,
            num_experts=4,
            tensors={"w": torch.empty(shape, dtype=torch.float32)},
        )

        pools = backend.configure_cache(
            cache_vram_mib=4,
            cache_vram_reserve_mib=0,
            stage_slots=2,
            event_factory=events.make,
            tensor_factory=_cpu_cache,
            staging_buffer_factory=_cpu_buffer,
            require_pinned_staging=False,
            free_bytes_provider=lambda: 4 * 1024 * 1024,
        )

        self.assertEqual(pools[0].cache_slots, 2)
        self.assertEqual(pools[1].cache_slots, 2)
        assert backend.staging_ring is not None
        self.assertEqual(backend.staging_ring.allocated_bytes, 2 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as directory:
            stats_path = Path(directory) / "stats.json"
            backend.flush_stats(stats_path)
            self.assertIn('"cache_slots": 2', stats_path.read_text())
        backend.close()

    def test_stats_flush_counts_only_successful_releases(self):
        events = _Events()
        backend = CpuMemoryExpertBackend()
        backend.configure_stats("ignored.json", 2)
        flushed = []
        backend.flush_stats = lambda path: flushed.append(path)
        host = _host_layer()
        ring = PinnedStagingRing(
            1,
            host.staging_payload_bytes,
            event_factory=events.make,
            buffer_factory=_cpu_buffer,
            require_pinned=False,
        )
        pool = ExpertCachePool(
            host,
            2,
            ring,
            event_factory=events.make,
            tensor_factory=_cpu_cache,
            route_complete_callback=backend._record_completed_route,
        )
        transfer, compute = _FakeStream(), _FakeStream()
        first = pool.acquire([0], transfer, compute)
        first.release_after(compute)
        self.assertEqual(flushed, [])
        with self.assertRaisesRegex(RuntimeError, "stale or double"):
            first.release_after(compute)
        self.assertEqual(flushed, [])
        second = pool.acquire([1], transfer, compute)
        second.release_after(compute)
        self.assertEqual(flushed, ["ignored.json"])

    def test_snapshot_reports_allocations_counters_and_completed_timing(self):
        backend = CpuMemoryExpertBackend()
        backend.register_host_layer(
            layer_id=0,
            top_k=1,
            num_experts=2,
            tensors={"w": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        )
        backend.configure_cache(
            cache_vram_mib=1,
            cache_vram_reserve_mib=0,
            stage_slots=1,
            event_factory=_FakeEvent,
            timing_event_factory=_TimingEvent,
            tensor_factory=_cpu_cache,
            staging_buffer_factory=_cpu_buffer,
            require_pinned_staging=False,
            free_bytes_provider=lambda: 1024 * 1024,
        )
        pool = backend.pools[0]
        transfer, compute = _FakeStream(), _FakeStream()
        first = pool.acquire([0], transfer, compute)
        first.release_after(compute)
        # The single staging slot forces a host synchronization before this
        # second pageable-to-pinned overwrite.
        second = pool.acquire([1], transfer, compute)
        second.release_after(compute)
        backend.record_moe_call()
        backend.record_microbatch()

        snapshot = backend.snapshot_stats()
        self.assertEqual(snapshot["totals"]["host_expert_bytes"], 16)
        self.assertEqual(snapshot["totals"]["cache_allocated_bytes"], 16)
        self.assertEqual(snapshot["cache_budget"]["actual_bytes"], 16)
        self.assertEqual(snapshot["totals"]["moe_calls"], 1)
        self.assertEqual(snapshot["totals"]["microbatches"], 1)
        self.assertEqual(snapshot["totals"]["staging_wait_count"], 1)
        self.assertEqual(snapshot["pools"]["0"]["hits"], 0)
        self.assertEqual(snapshot["pools"]["0"]["misses"], 2)
        self.assertEqual(snapshot["pools"]["0"]["h2d_bytes"], 16)
        self.assertEqual(snapshot["pools"]["0"]["h2d_time_ns"], 5_000_000)
        self.assertEqual(snapshot["pools"]["0"]["ready_wait_count"], 2)
        # Each ready event is made a single compute-stream dependency.  The
        # distinct timing pair records 2.5ms for each of those two waits.
        self.assertEqual(len(compute.waited_events), 2)
        self.assertEqual(snapshot["pools"]["0"]["ready_wait_time_ns"], 5_000_000)
        backend.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_staging_ring_uses_pinned_memory(self):
        host = HostExpertLayer.from_tensors(
            layer_id=0,
            top_k=1,
            num_experts=2,
            tensors={
                "w": torch.tensor([[1.0], [2.0]]),
                "s": torch.tensor([[3.0], [4.0]]),
            },
        )
        ring = PinnedStagingRing(
            1, host.staging_payload_bytes, event_factory=torch.cuda.Event
        )
        transfer = torch.cuda.Stream()
        pool = ExpertCachePool(host, 1, ring, event_factory=torch.cuda.Event)
        lease = pool.acquire([1], transfer, torch.cuda.current_stream())
        torch.cuda.synchronize()
        torch.testing.assert_close(
            pool.cache_tensors["w"][0], torch.tensor([2.0], device="cuda")
        )
        lease.release_after(torch.cuda.current_stream())
        replacement = pool.acquire([0], transfer, torch.cuda.current_stream())
        replacement.release_after(torch.cuda.current_stream())
        pool.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_pinned_source_bypasses_staging_ring(self):
        weights = torch.tensor([[1.0], [2.0]]).pin_memory()
        host = HostExpertLayer.from_tensors(
            layer_id=0,
            top_k=1,
            num_experts=2,
            tensors={"w": weights},
        )
        ring = PinnedStagingRing(
            1, host.staging_payload_bytes, event_factory=torch.cuda.Event
        )

        slot, staged = ring.stage(host, 1)

        self.assertIsNone(slot)
        self.assertTrue(staged["w"].is_pinned())
        self.assertEqual(staged["w"].data_ptr(), weights[1].data_ptr())
        self.assertEqual(host.pinned_bytes, host.total_bytes)
        ring.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_unquantized_final_layout_cache_matches_baseline_across_microbatches(
        self,
    ):
        """Exercise the actual pageable->pinned->slot path with unquantized experts.

        This intentionally uses a small reference MoE equation instead of a
        FusedMoE runner: runner construction varies by CUDA build, whereas the
        residency contract under test is that final expert tensors, routed IDs,
        and weighted outputs are unchanged when logical IDs are compacted to
        cache slots.  The real-model E2E test owns the kernel integration.
        """
        torch.manual_seed(7)
        num_experts, hidden_size, top_k, cache_slots = 4, 3, 2, 2
        # These are the post-load, final unquantized tensors a kernel would
        # consume.  They deliberately stay pageable on CPU for the repository.
        host_weights = torch.randn(num_experts, hidden_size, hidden_size)
        host_biases = torch.randn(num_experts, hidden_size)
        self.assertFalse(host_weights.is_pinned())
        host = HostExpertLayer.from_tensors(
            layer_id=3,
            top_k=top_k,
            num_experts=num_experts,
            tensors={"weight": host_weights, "bias": host_biases},
        )
        ring = PinnedStagingRing(
            1,
            host.staging_payload_bytes,
            event_factory=torch.cuda.Event,
        )
        pool = ExpertCachePool(host, cache_slots, ring, event_factory=torch.cuda.Event)
        transfer = torch.cuda.Stream()
        compute = torch.cuda.Stream()

        tokens_cpu = torch.randn(4, hidden_size)
        route_ids_cpu = torch.tensor([[0, 1], [1, 0], [2, 3], [2, 3]])
        route_weights_cpu = torch.tensor(
            [[0.75, 0.25], [0.20, 0.80], [0.40, 0.60], [0.50, 0.50]]
        )

        def reference_moe(tokens, expert_ids, expert_weights, weights, biases):
            result = torch.zeros_like(tokens)
            for token_index, (ids, coefficients) in enumerate(
                zip(expert_ids.tolist(), expert_weights)
            ):
                for expert_id, coefficient in zip(ids, coefficients):
                    result[token_index] += coefficient * (
                        tokens[token_index] @ weights[expert_id].T + biases[expert_id]
                    )
            return result

        expected = reference_moe(
            tokens_cpu, route_ids_cpu, route_weights_cpu, host_weights, host_biases
        )
        outputs = []
        # The first range exercises duplicate routes in one lease, the second
        # forces two evictions, and the third observes cache hits for 2 and 3.
        for start, end in ((0, 2), (2, 3), (3, 4)):
            ids = route_ids_cpu[start:end]
            coefficients = route_weights_cpu[start:end].to("cuda")
            tokens = tokens_cpu[start:end].to("cuda")
            lease = pool.acquire(ids.reshape(-1).tolist(), transfer, compute)
            with torch.cuda.stream(compute):
                remapped_ids = ids.clone()
                for logical, physical in lease.logical_to_physical.items():
                    remapped_ids[ids == logical] = physical
                routed_output = reference_moe(
                    tokens,
                    remapped_ids,
                    coefficients,
                    pool.cache_tensors["weight"],
                    pool.cache_tensors["bias"],
                )
            lease.release_after(compute)
            outputs.append(routed_output)

        torch.cuda.synchronize()
        torch.testing.assert_close(
            torch.cat(outputs).cpu(), expected, rtol=1e-5, atol=1e-6
        )
        self.assertEqual(pool.cache_tensors["weight"].shape[0], cache_slots)
        self.assertEqual(pool.cache_tensors["bias"].shape[0], cache_slots)
        self.assertNotEqual(pool.cache_tensors["weight"].shape[0], num_experts)
        self.assertEqual(pool.stats["cache_misses"], 4)
        self.assertEqual(pool.stats["cache_hits"], 2)
        self.assertEqual(pool.stats["evictions"], 2)
        self.assertEqual(pool.stats["requested_experts"], 6)
        # No expert-major cache allocation may scale to the full repository.
        self.assertEqual(pool.allocated_bytes, cache_slots * host.expert_tensor_bytes)
        self.assertLess(pool.allocated_bytes, host.total_bytes)
        pool.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_awq_final_layout_payload_is_bit_exact_in_compact_slots(self):
        """Copy already-repacked AWQ/Marlin payloads without reinterpreting them.

        AWQ's checkpoint-to-Marlin conversion belongs to the baseline post-load
        code and is independently covered by the AWQ repack suite.  This test
        starts after that conversion and proves the CPU backend only transports
        every final-layout tensor byte-for-byte into compact cache slots.
        """
        num_experts, cache_slots = 3, 2
        final_layout = {
            "w13_qweight": torch.arange(3 * 2 * 4, dtype=torch.int32).reshape(3, 2, 4),
            "w13_scales": torch.arange(3 * 2 * 2, dtype=torch.float16).reshape(3, 2, 2),
            "w13_qzeros": torch.arange(3 * 2, dtype=torch.int32).reshape(3, 2),
            "w2_qweight": torch.arange(3 * 4 * 2, dtype=torch.int32).reshape(3, 4, 2),
            "w2_scales": torch.arange(3 * 2, dtype=torch.float16).reshape(3, 2),
            "w2_qzeros": torch.arange(3 * 2, dtype=torch.int32).reshape(3, 2),
        }
        host = HostExpertLayer.from_tensors(
            layer_id=4,
            top_k=2,
            num_experts=num_experts,
            tensors=final_layout,
        )
        ring = PinnedStagingRing(
            1, host.staging_payload_bytes, event_factory=torch.cuda.Event
        )
        pool = ExpertCachePool(host, cache_slots, ring, event_factory=torch.cuda.Event)
        transfer = torch.cuda.Stream()
        compute = torch.cuda.Stream()
        lease = pool.acquire([2, 0, 2], transfer, compute)
        torch.cuda.synchronize()

        self.assertEqual(lease.logical_to_physical, {2: 0, 0: 1})
        for name, source in final_layout.items():
            for logical, physical in lease.logical_to_physical.items():
                self.assertTrue(
                    torch.equal(
                        pool.cache_tensors[name][physical].cpu(), source[logical]
                    ),
                    msg=f"{name} expert {logical} did not preserve final layout",
                )
            self.assertEqual(pool.cache_tensors[name].shape[0], cache_slots)
        self.assertLess(pool.allocated_bytes, host.total_bytes)
        lease.release_after(compute)
        pool.close()


if __name__ == "__main__":
    unittest.main()
