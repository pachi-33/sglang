import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.qwen3_5.expert_pack.format import (
    ALIGNMENT,
    COMPONENTS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPERT_LAYERS,
    EXPERTS_PER_LAYER,
    FORMAT_ID,
    PACK_FILENAME,
    PACK_SIZE,
    PAYLOAD_SIZE,
    RECORD_COUNT,
    RECORD_STRIDE,
)
from sglang.srt.layers.qwen3_5.expert_pack.store import (
    ExpertCacheBusyError,
    ExpertOffloadConfig,
    ExpertPackFailedError,
    ExpertPackIntegrityError,
    ExpertPackStore,
    _CachePolicy,
    _Placement,
)


class TestExpertOffloadConfig(unittest.TestCase):
    def test_cache_budget_uses_exact_payload_floor(self):
        config = ExpertOffloadConfig("manifest.json", cache_mib=7168)
        self.assertEqual(config.cache_capacity, 4247)
        self.assertEqual(config.cache_bytes, 4247 * PAYLOAD_SIZE)

    def test_rejects_less_than_one_layer_working_set(self):
        with self.assertRaisesRegex(ValueError, "worst-case layer"):
            ExpertOffloadConfig("manifest.json", cache_mib=432)
        self.assertEqual(
            ExpertOffloadConfig("manifest.json", cache_mib=433).cache_capacity,
            256,
        )

    def test_rejects_more_workers_than_staging_buffers(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            ExpertOffloadConfig("manifest.json", stage_slots=1, io_workers=2)


class TestCachePolicy(unittest.TestCase):
    def test_plan_does_not_publish_before_commit(self):
        policy = _CachePolicy(2)
        placement = policy.plan(((1, 7),))
        self.assertEqual(policy.resident, 0)
        self.assertNotIn((1, 7), policy.key_to_slot)
        assignments = policy.commit_and_lease(placement)
        self.assertEqual(policy.key_to_slot[(1, 7)], assignments[0][0])

    def test_lfu_then_lru_victim_order(self):
        policy = _CachePolicy(2)
        initial = policy.commit_and_lease(policy.plan(((1, 1), (1, 2))))
        policy.release(initial)

        hot = policy.commit_and_lease(policy.plan(((1, 1),)))
        policy.release(hot)
        replacement = policy.plan(((1, 3),))
        self.assertEqual(replacement[0].prior_key, (1, 2))

        installed = policy.commit_and_lease(replacement)
        policy.release(installed)
        self.assertEqual(set(policy.key_to_slot), {(1, 1), (1, 3)})

    def test_active_lease_protects_slot(self):
        policy = _CachePolicy(1)
        policy.commit_and_lease(policy.plan(((1, 1),)))
        with self.assertRaises(ExpertCacheBusyError):
            policy.plan(((1, 2),))

    def test_stale_epoch_release_is_rejected(self):
        policy = _CachePolicy(1)
        old = policy.commit_and_lease(policy.plan(((1, 1),)))
        policy.release(old)
        current = policy.commit_and_lease(policy.plan(((1, 2),)))
        with self.assertRaisesRegex(RuntimeError, "stale expert lease"):
            policy.release(old)
        policy.release(current)


class TestBoundedRecordRead(unittest.TestCase):
    def _store_with_fd(self, path: Path):
        store = object.__new__(ExpertPackStore)
        store._fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, store._fd)
        return store

    def test_reads_exact_payload_and_checks_sha256(self):
        payload = bytes((index % 251 for index in range(PAYLOAD_SIZE)))
        prefix = b"header"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "experts.pack")
            path.write_bytes(prefix + payload + b"padding")
            store = self._store_with_fd(path)
            record = SimpleNamespace(
                offset=len(prefix),
                payload_size=PAYLOAD_SIZE,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            staging = torch.empty(PAYLOAD_SIZE, dtype=torch.uint8)
            result = store._read_payload(record, staging)
        self.assertEqual(staging.numpy().tobytes(), payload)
        self.assertGreaterEqual(result.read_ns, 0)
        self.assertGreaterEqual(result.checksum_ns, 0)

    def test_short_read_and_checksum_mismatch_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            short_path = Path(directory, "short.pack")
            short_path.write_bytes(b"short")
            short_store = self._store_with_fd(short_path)
            staging = torch.empty(PAYLOAD_SIZE, dtype=torch.uint8)
            record = SimpleNamespace(
                offset=0,
                payload_size=PAYLOAD_SIZE,
                sha256="0" * 64,
            )
            with self.assertRaisesRegex(OSError, "short ExpertPack read"):
                short_store._read_payload(record, staging)

            full_path = Path(directory, "full.pack")
            full_path.write_bytes(bytes(PAYLOAD_SIZE))
            full_store = self._store_with_fd(full_path)
            with self.assertRaises(ExpertPackIntegrityError):
                full_store._read_payload(record, staging)


class TestStoreStartupValidation(unittest.TestCase):
    def test_rejects_pack_whose_size_differs_from_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest_path = directory / "manifest.json"
            manifest_path.write_text("{}")
            pack_path = directory / "experts.pack"
            pack_path.write_bytes(b"truncated")
            manifest = SimpleNamespace(
                pack_size=123,
                pack_path=lambda _: pack_path,
            )
            with patch(
                "sglang.srt.layers.qwen3_5.expert_pack.store.load_manifest",
                return_value=manifest,
            ):
                with self.assertRaisesRegex(ValueError, "file size"):
                    ExpertPackStore(ExpertOffloadConfig(manifest_path, cache_mib=433))


class _ImmediateFuture:
    def __init__(self, function, args, kwargs):
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def result(self):
        return self._function(*self._args, **self._kwargs)


class _ImmediateExecutor:
    def submit(self, function, *args, **kwargs):
        return _ImmediateFuture(function, args, kwargs)

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


class _FakeEvent:
    def record(self, stream):
        pass

    def synchronize(self):
        pass


class TestStoreFailureLatching(unittest.TestCase):
    """Exercise the complete store state machine without touching CUDA."""

    def _make_ready_store(self, directory: Path) -> ExpertPackStore:
        manifest_path = directory / "manifest.json"
        manifest_path.write_text("{}")
        pack_path = directory / PACK_FILENAME
        pack_path.write_bytes(b"")
        record = SimpleNamespace(
            offset=0,
            payload_size=PAYLOAD_SIZE,
            sha256=hashlib.sha256(bytes(PAYLOAD_SIZE)).hexdigest(),
        )
        manifest = SimpleNamespace(
            pack_size=0,
            pack_path=lambda _: pack_path,
            record=lambda layer_id, expert_id: record,
        )
        with patch(
            "sglang.srt.layers.qwen3_5.expert_pack.store.load_manifest",
            return_value=manifest,
        ):
            store = ExpertPackStore(
                ExpertOffloadConfig(
                    manifest_path,
                    cache_mib=433,
                    stage_slots=1,
                    io_workers=1,
                )
            )
        store._state = "READY"
        store._device = torch.device("cpu")
        store._transfer_stream = SimpleNamespace()
        store._executor = _ImmediateExecutor()
        store._staging = [torch.empty(PAYLOAD_SIZE, dtype=torch.uint8)]
        store._staging_ready = [None]

        def close_without_cuda():
            store._device = None
            store.close()

        self.addCleanup(close_without_cuda)
        return store

    def _assert_failed_latch(
        self, store: ExpertPackStore, ids: torch.Tensor, category: str
    ) -> None:
        self.assertEqual(store.state, "FAILED")
        self.assertIsNotNone(store.failure)
        self.assertEqual(store._stats[category], 1)
        self.assertEqual(store._stats["fatal_errors"], 1)
        original_failure = store.failure
        with self.assertRaisesRegex(ExpertPackFailedError, "requires process restart"):
            store.acquire(1, ids)
        self.assertIs(store.failure, original_failure)
        self.assertEqual(store._stats["fatal_errors"], 1)

    def test_short_read_does_not_publish_and_latches_io_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._make_ready_store(Path(directory))
            ids = torch.full((1, 8), 7, dtype=torch.int32)
            with patch.object(
                store, "_read_payload", side_effect=OSError("short ExpertPack read")
            ):
                with self.assertRaisesRegex(OSError, "short ExpertPack read"):
                    store.acquire(1, ids)
            self.assertEqual(store._policy.key_to_slot, {})
            self.assertEqual(store._policy.resident, 0)
            self._assert_failed_latch(store, ids, "io_errors")

    def test_nth_short_read_discards_earlier_h2d_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._make_ready_store(Path(directory))
            ids = torch.tensor([[7, 8, 7, 8, 7, 8, 7, 8]], dtype=torch.int32)
            first_read = SimpleNamespace(read_ns=3, checksum_ns=5)
            with patch.object(
                store,
                "_read_payload",
                side_effect=(first_read, OSError("short ExpertPack read N=2")),
            ), patch.object(
                store, "_enqueue_install", return_value=_FakeEvent()
            ) as enqueue:
                with self.assertRaisesRegex(OSError, "N=2"):
                    store.acquire(1, ids)
            enqueue.assert_called_once()
            self.assertEqual(store._stats["h2d_bytes"], PAYLOAD_SIZE)
            self.assertEqual(store._policy.key_to_slot, {})
            self.assertEqual(store._policy.resident, 0)
            self._assert_failed_latch(store, ids, "io_errors")

    def test_checksum_failure_does_not_publish_and_latches_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._make_ready_store(Path(directory))
            ids = torch.full((1, 8), 9, dtype=torch.int32)
            with patch.object(
                store,
                "_read_payload",
                side_effect=ExpertPackIntegrityError("checksum mismatch"),
            ):
                with self.assertRaisesRegex(
                    ExpertPackIntegrityError, "checksum mismatch"
                ):
                    store.acquire(1, ids)
            self.assertEqual(store._policy.key_to_slot, {})
            self.assertEqual(store._policy.resident, 0)
            self._assert_failed_latch(store, ids, "checksum_errors")

    def test_illegal_mapping_is_rejected_before_read_and_latches_epoch_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._make_ready_store(Path(directory))
            ids = torch.full((1, 8), 11, dtype=torch.int32)
            illegal = (
                _Placement(
                    key=(1, 11),
                    slot=store.cache_capacity,
                    hit=False,
                    prior_key=None,
                    prior_epoch=0,
                ),
            )
            with patch.object(
                store._policy, "plan", return_value=illegal
            ), patch.object(store, "_read_payload") as read_payload:
                with self.assertRaisesRegex(
                    ExpertPackFailedError, "illegal slot mapping"
                ):
                    store.acquire(1, ids)
            read_payload.assert_not_called()
            self.assertEqual(store._policy.key_to_slot, {})
            self.assertEqual(store._policy.resident, 0)
            self._assert_failed_latch(store, ids, "epoch_errors")

    def test_illegal_committed_assignment_never_becomes_cuda_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._make_ready_store(Path(directory))
            ids = torch.full((1, 8), 13, dtype=torch.int32)
            read_result = SimpleNamespace(read_ns=0, checksum_ns=0)
            illegal_assignment = ((store.cache_capacity, 1),)
            with patch.object(
                store, "_read_payload", return_value=read_result
            ), patch.object(
                store, "_enqueue_install", return_value=_FakeEvent()
            ), patch.object(
                store._policy,
                "commit_and_lease",
                return_value=illegal_assignment,
            ):
                with self.assertRaisesRegex(
                    ExpertPackFailedError, "illegal assignment"
                ):
                    store.acquire(1, ids)
            self.assertEqual(store._policy.key_to_slot, {})
            self.assertEqual(store._policy.resident, 0)
            self._assert_failed_latch(store, ids, "epoch_errors")

    def test_stale_epoch_release_latches_failure_and_preserves_current_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._make_ready_store(Path(directory))
            store._policy = _CachePolicy(1)
            old = store._policy.commit_and_lease(store._policy.plan(((1, 3),)))
            store._policy.release(old)
            current = store._policy.commit_and_lease(store._policy.plan(((1, 5),)))
            self.assertNotEqual(old[0][1], current[0][1])
            with patch(
                "sglang.srt.layers.qwen3_5.expert_pack.store.torch.cuda.Event",
                return_value=_FakeEvent(),
            ):
                with self.assertRaisesRegex(RuntimeError, "stale expert lease"):
                    store._release(old, SimpleNamespace())
            self.assertEqual(store._policy.slots[0].key, (1, 5))
            self.assertEqual(store._policy.slots[0].leases, 1)
            ids = torch.full((1, 8), 5, dtype=torch.int32)
            self._assert_failed_latch(store, ids, "epoch_errors")


class TestExpertPackStoreCUDA(V100TestCase):
    """A small data-path test; its pack is sparse and its cache is 433 MiB."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.cuda.get_device_capability() != (7, 0):
            raise unittest.SkipTest("ExpertPack CUDA store is V100-only")

    @staticmethod
    def _write_sparse_pack(directory: Path) -> Path:
        payload_digest = hashlib.sha256(bytes(PAYLOAD_SIZE)).hexdigest()
        records = []
        for index in range(RECORD_COUNT):
            layer_delta, expert_id = divmod(index, EXPERTS_PER_LAYER)
            records.append(
                {
                    "layer_id": EXPERT_LAYERS[0] + layer_delta,
                    "expert_id": expert_id,
                    "offset": index * RECORD_STRIDE,
                    "payload_size": PAYLOAD_SIZE,
                    "sha256": payload_digest,
                }
            )
        manifest = {
            "format": FORMAT_ID,
            "complete": True,
            "pack_file": PACK_FILENAME,
            "pack_size": PACK_SIZE,
            "pack_sha256": "0" * 64,
            "record_count": RECORD_COUNT,
            "record_stride": RECORD_STRIDE,
            "payload_size": PAYLOAD_SIZE,
            "alignment": ALIGNMENT,
            "model": {
                "num_layers": 40,
                "num_experts": 256,
                "top_k": 8,
                "hidden_size": 2048,
                "intermediate_size": 512,
                "offloaded_layers": list(EXPERT_LAYERS),
                "quantization": "nvfp4",
            },
            "components": {name: spec.to_dict() for name, spec in COMPONENTS.items()},
            "source": {
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "index_sha256": EXPECTED_INDEX_SHA256,
                "shards": [{"file": "synthetic.safetensors", "size": 1}],
            },
            "records": records,
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        with (directory / PACK_FILENAME).open("wb") as handle:
            handle.truncate(PACK_SIZE)
        return manifest_path

    def test_typed_cache_events_and_hot_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_sparse_pack(Path(directory))
            config = ExpertOffloadConfig(
                manifest_path,
                cache_mib=433,
                stage_slots=2,
                io_workers=1,
            )
            with ExpertPackStore(config, device="cuda") as store:
                ids = torch.full((1, 8), 7, dtype=torch.int32, device="cuda")
                with store.acquire(1, ids) as lease:
                    self.assertEqual(lease.gate_up.data.shape, (256, 1024, 1024))
                    self.assertEqual(lease.down.data.shape, (256, 2048, 256))
                    self.assertEqual(lease.expert_to_slot.shape, (256,))
                    self.assertEqual(lease.first_slot, 0)
                    self.assertEqual(lease.expert_to_slot[7].item(), 0)
                    self.assertEqual(lease.expert_to_slot[0].item(), -1)
                with store.acquire(1, ids):
                    pass
                stats = store.snapshot()
                self.assertEqual(stats["pack_reads"], 1)
                self.assertEqual(stats["cache_misses"], 1)
                self.assertEqual(stats["cache_hits"], 1)
                self.assertEqual(stats["h2d_bytes"], PAYLOAD_SIZE)
                store.fail(RuntimeError("asynchronous kernel failure"))
                self.assertEqual(store.state, "FAILED")
                with self.assertRaisesRegex(RuntimeError, "requires process restart"):
                    store.acquire(1, ids)


if __name__ == "__main__":
    unittest.main()
