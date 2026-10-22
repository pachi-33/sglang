"""Cache statistics use received counts; native residency owns mapping and snapshots."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sglang.srt.offloading.moe.cache_metadata import CacheRouteStats
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCacheRouteStats(unittest.TestCase):
    def test_repeated_assignments_across_disjoint_waves(self):
        stats = CacheRouteStats.allocate("cpu")
        for counts, hits in (([2, 3], [True, False]), ([0, 5], [False, True])):
            stats.record_counts(
                torch.tensor(counts), torch.tensor(hits), torch.tensor(True)
            )
        self.assertEqual(
            (int(stats.accesses), int(stats.hits), int(stats.misses)), (10, 7, 3)
        )
        pointers = [item.storage_key for item in stats.allocations()]
        stats.clear()
        self.assertEqual(
            (int(stats.accesses), int(stats.hits), int(stats.misses)), (0, 0, 0)
        )
        self.assertEqual([item.storage_key for item in stats.allocations()], pointers)

    def test_disabled_and_empty_demand_leave_counters_unchanged(self):
        stats = CacheRouteStats.allocate("cpu")
        for counts, enabled in (([9, 9], False), ([0, 0], True), ([], True)):
            stats.record_counts(
                torch.tensor(counts, dtype=torch.int64),
                torch.zeros(len(counts), dtype=torch.bool),
                torch.tensor(enabled),
            )
        self.assertEqual(int(stats.accesses + stats.hits + stats.misses), 0)

    def test_all_hits_and_zero_cache(self):
        for hit, expected in ((True, (7, 7, 0)), (False, (7, 0, 7))):
            stats = CacheRouteStats.allocate("cpu")
            stats.record_counts(
                torch.tensor([3, 4]), torch.full((2,), hit), torch.tensor(True)
            )
            self.assertEqual(
                (int(stats.accesses), int(stats.hits), int(stats.misses)), expected
            )

    def test_counting_does_not_read_device_values(self):
        stats = CacheRouteStats.allocate("cpu")
        with ExitStack() as stack:
            for name in ("item", "tolist", "cpu", "__bool__"):
                stack.enter_context(
                    patch.object(torch.Tensor, name, side_effect=AssertionError(name))
                )
            stats.record_counts(
                torch.tensor([3, 4]), torch.tensor([True, False]), torch.tensor(True)
            )
        self.assertEqual(int(stats.hits), 3)


if __name__ == "__main__":
    unittest.main()
