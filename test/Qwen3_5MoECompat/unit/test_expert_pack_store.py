import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.qwen3_5.expert_pack.format import PAYLOAD_SIZE
from sglang.srt.layers.qwen3_5.expert_pack.store import (
    ExpertCacheBusyError,
    ExpertOffloadConfig,
    ExpertPackIntegrityError,
    ExpertPackStore,
    PinnedExpertRepository,
    _CachePolicy,
)


class TestExpertOffloadConfig(unittest.TestCase):
    def test_default_ratio_defines_layer_and_scratch_capacity(self):
        config = ExpertOffloadConfig("manifest.json")
        self.assertEqual(config.layer_capacity, 102)
        self.assertEqual(config.scratch_capacity, 154)
        self.assertEqual(config.cache_capacity, 4030)
        self.assertEqual(config.cache_bytes, 4030 * PAYLOAD_SIZE)
        self.assertEqual(config.realized_cache_ratio, 102 / 256)

    def test_mib_is_only_a_safety_ceiling(self):
        with self.assertRaisesRegex(ValueError, "safety limit"):
            ExpertOffloadConfig("manifest.json", cache_ratio=0.4, cache_mib=6800)
        self.assertEqual(
            ExpertOffloadConfig(
                "manifest.json", cache_ratio=0.4, cache_mib=6801
            ).cache_capacity,
            4030,
        )

    def test_minimum_ratio_still_has_full_prefill_working_set(self):
        config = ExpertOffloadConfig(
            "manifest.json", cache_ratio=8 / 256, cache_mib=1024
        )
        self.assertEqual(config.layer_capacity, 8)
        self.assertEqual(config.scratch_capacity, 248)
        self.assertEqual(config.layer_capacity + config.scratch_capacity, 256)
        with self.assertRaisesRegex(ValueError, "cache_ratio"):
            ExpertOffloadConfig("manifest.json", cache_ratio=0.01)

    def test_rejects_more_workers_than_load_queue(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            ExpertOffloadConfig("manifest.json", stage_slots=1, io_workers=2)


class TestLayerLocalStrictLRU(unittest.TestCase):
    def _install(self, policy, keys, **kwargs):
        assignments = policy.commit_and_lease(policy.plan(keys), **kwargs)
        if kwargs.get("lease", True):
            policy.release(assignments)
        return assignments

    def test_demand_hit_refreshes_lru(self):
        policy = _CachePolicy(2, layer_id=1)
        self._install(policy, ((1, 1), (1, 2)))
        self._install(policy, ((1, 1),))
        victim = policy.plan(((1, 3),))[0]
        self.assertEqual(victim.prior_key, (1, 2))

    def test_prefetch_hit_does_not_refresh_lru_but_insert_does(self):
        policy = _CachePolicy(2, layer_id=1)
        self._install(policy, ((1, 1), (1, 2)))
        policy.commit_and_lease(policy.plan(((1, 1),)), lease=False, touch_hits=False)
        replacement = policy.plan(((1, 3),))
        self.assertEqual(replacement[0].prior_key, (1, 1))
        policy.commit_and_lease(replacement, lease=False, touch_hits=False)
        self.assertEqual(set(policy.key_to_slot), {(1, 2), (1, 3)})

    def test_layer_identity_and_slot_interval_are_enforced(self):
        layer1 = _CachePolicy(1, slot_offset=0, layer_id=1)
        layer2 = _CachePolicy(1, slot_offset=1, layer_id=2)
        self._install(layer1, ((1, 7),))
        self._install(layer2, ((2, 8),))
        self.assertEqual(layer1.key_to_slot[(1, 7)], 0)
        self.assertEqual(layer2.key_to_slot[(2, 8)], 1)
        with self.assertRaisesRegex(ValueError, "another layer"):
            layer1.plan(((2, 9),))

    def test_active_lease_and_clear_are_safe(self):
        policy = _CachePolicy(1, layer_id=1)
        assignment = policy.commit_and_lease(policy.plan(((1, 1),)))
        with self.assertRaises(ExpertCacheBusyError):
            policy.plan(((1, 2),))
        with self.assertRaises(ExpertCacheBusyError):
            policy.clear()
        old_epoch = assignment[0][1]
        policy.release(assignment)
        policy.clear()
        replacement = policy.commit_and_lease(policy.plan(((1, 2),)))
        self.assertGreater(replacement[0][1], old_epoch)

    def test_prefill_lru_selection_uses_last_token_rank_accesses(self):
        values = (1, 2, 3, 1, 4, 2)
        self.assertEqual(ExpertPackStore._final_lru(values, 3), (1, 4, 2))


class TestPinnedExpertRepository(unittest.TestCase):
    def test_direct_payload_read_and_checksum(self):
        payload = bytes(index % 251 for index in range(PAYLOAD_SIZE))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "experts.pack")
            path.write_bytes(b"prefix" + payload)
            record = SimpleNamespace(
                offset=6,
                payload_size=PAYLOAD_SIZE,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            manifest = SimpleNamespace(record=lambda layer, expert: record)
            repository = PinnedExpertRepository(
                manifest, path, io_workers=1, queue_depth=1
            )
            repository.layers[1] = torch.empty((1, PAYLOAD_SIZE), dtype=torch.uint8)
            fd = os.open(path, os.O_RDONLY)
            try:
                result = repository._read_one(fd, 1, 0)
            finally:
                os.close(fd)
            self.assertEqual(repository.layers[1][0].numpy().tobytes(), payload)
            self.assertGreaterEqual(result.read_ns, 0)

    def test_checksum_error_is_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "experts.pack")
            path.write_bytes(bytes(PAYLOAD_SIZE))
            record = SimpleNamespace(
                offset=0, payload_size=PAYLOAD_SIZE, sha256="f" * 64
            )
            repository = PinnedExpertRepository(
                SimpleNamespace(record=lambda layer, expert: record),
                path,
                io_workers=1,
                queue_depth=1,
            )
            repository.layers[1] = torch.empty((1, PAYLOAD_SIZE), dtype=torch.uint8)
            fd = os.open(path, os.O_RDONLY)
            try:
                with self.assertRaises(ExpertPackIntegrityError):
                    repository._read_one(fd, 1, 0)
            finally:
                os.close(fd)

    def test_store_rejects_pack_size_before_allocating_pinned_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "experts.pack")
            path.write_bytes(b"short")
            manifest = SimpleNamespace(
                pack_size=100,
                pack_path=lambda _: path,
            )
            with patch(
                "sglang.srt.layers.qwen3_5.expert_pack.store.load_manifest",
                return_value=manifest,
            ), self.assertRaisesRegex(ValueError, "file size"):
                ExpertPackStore(ExpertOffloadConfig(Path(directory, "manifest.json")))


if __name__ == "__main__":
    unittest.main()
