import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import sglang.srt.moe_trace.writer as writer_module
from sglang.srt.moe_trace.types import (
    MoeTraceBatchOutput,
    MoeTraceSite,
    MoeTraceSiteLayout,
)
from sglang.srt.moe_trace.writer import MoeTraceWriter, safe_request_id


def _layouts():
    return [
        MoeTraceSiteLayout(
            MoeTraceSite(3, "layers.0.mlp", 0, "topk", 3, 8, 2),
            activation_q_offset=0,
            activation_q_width=3,
            activation_scale_offset=0,
            activation_scale_width=1,
            route_offset=0,
        ),
        MoeTraceSiteLayout(
            MoeTraceSite(9, "layers.1.mlp", 1, "topk", 2, 16, 1),
            activation_q_offset=3,
            activation_q_width=2,
            activation_scale_offset=1,
            activation_scale_width=2,
            route_offset=2,
        ),
    ]


def _payload(request_ids, release=None, *, activations=True, routes=True):
    rows = len(request_ids)
    return MoeTraceBatchOutput(
        request_ids=request_ids,
        input_token_ids=torch.arange(rows, dtype=torch.int64),
        positions=torch.arange(10, 10 + rows, dtype=torch.int64),
        site_valid=torch.ones(rows, 2, dtype=torch.bool),
        activation_q=(
            torch.arange(rows * 5, dtype=torch.uint8).reshape(rows, 5)
            if activations
            else None
        ),
        activation_scales=(
            torch.ones(rows, 3, dtype=torch.float16) if activations else None
        ),
        expert_ids=(
            torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3)
            if routes
            else None
        ),
        expert_weights=torch.ones(rows, 3, dtype=torch.float32) if routes else None,
        release_callback=release,
    )


def _manifest(root, request_id):
    return json.loads(
        (root / safe_request_id(request_id) / "manifest.json").read_text()
    )


class TestMoeTraceWriter(unittest.TestCase):
    def test_safe_request_id_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            from pathlib import Path

            root = Path(temp)
            request_id = "../../outside/..\\also"
            name = safe_request_id(request_id)
            self.assertNotIn("/", name)
            self.assertNotIn("\\", name)
            self.assertNotIn("..", name)
            writer = MoeTraceWriter(root, _layouts())
            writer.submit(_payload([request_id]))
            writer.finalize_request(request_id)
            writer.close()
            self.assertTrue(
                (root / name / "rank-000" / "chunk-000000.safetensors").exists()
            )
            self.assertFalse((root.parent / "outside").exists())

    def test_splits_requests_and_optional_features(self):
        from pathlib import Path

        from safetensors.torch import load_file

        for activations, routes in ((False, True), (True, False), (True, True)):
            with self.subTest(
                activations=activations, routes=routes
            ), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                writer = MoeTraceWriter(
                    root,
                    _layouts(),
                    quantization=(
                        {"scheme": "symmetric-groupwise-int4", "group_size": 2}
                        if activations
                        else {}
                    ),
                )
                writer.submit(
                    _payload(
                        ["one", "two", "one"], activations=activations, routes=routes
                    )
                )
                writer.finalize_request("one")
                writer.finalize_request("two")
                writer.close()
                one, two = _manifest(root, "one"), _manifest(root, "two")
                self.assertEqual((one["total_rows"], two["total_rows"]), (2, 1))
                self.assertIs(one["feature_flags"]["activations"], activations)
                self.assertIs(one["feature_flags"]["routes"], routes)
                self.assertIs(bool(one["quantization"]), activations)
                keys = set(
                    load_file(
                        str(root / safe_request_id("one") / one["chunks"][0]["path"])
                    )
                )
                self.assertIs("site.3.activation_q" in keys, activations)
                self.assertIs("site.3.expert_ids" in keys, routes)
                self.assertFalse(list(root.rglob("*.partial")))

    def test_release_once_for_written_and_dropped_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            released = []
            writer = MoeTraceWriter(temp, _layouts(), queue_depth=1, overflow="drop")
            original = writer._write_payload

            def slow(payload):
                time.sleep(0.1)
                original(payload)

            writer._write_payload = slow
            self.assertTrue(
                writer.submit(_payload(["a"], lambda: released.append("written")))
            )
            callbacks = []
            dropped = False
            for number in range(10):
                callback = lambda number=number: callbacks.append(number)
                if not writer.submit(_payload(["b"], callback)):
                    dropped = True
                    break
            writer.close()
            self.assertTrue(dropped)
            self.assertEqual(released.count("written"), 1)
            self.assertEqual(len(callbacks), len(set(callbacks)))
            self.assertEqual(writer.stats["dropped_rows"], 1)

    def test_worker_failure_is_persisted_and_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            released = []
            writer = MoeTraceWriter(temp, _layouts())
            writer._write_chunk = lambda *_: (_ for _ in ()).throw(OSError("disk full"))
            writer.submit(_payload(["x"], lambda: released.append(1)))
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                writer.close()
            self.assertEqual(released, [1])
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                writer.submit(_payload(["x"]))

    def test_failed_atomic_chunk_leaves_no_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def broken_save(_tensors, path):
                Path(path).write_bytes(b"incomplete")
                raise OSError("disk full")

            with patch.object(
                writer_module, "_require_safetensors", return_value=broken_save
            ):
                writer = MoeTraceWriter(root, _layouts())
                writer.submit(_payload(["x"]))
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    writer.close()
            self.assertFalse(list(root.rglob("*.partial")))

    def test_block_never_drops(self):
        with tempfile.TemporaryDirectory() as temp:
            released = []
            writer = MoeTraceWriter(temp, _layouts(), queue_depth=1, overflow="block")
            writer.submit(_payload(["x"], lambda: released.append(1)))
            writer.submit(_payload(["x"], lambda: released.append(2)))
            writer.close()
            self.assertEqual(writer.stats["dropped_rows"], 0)
            self.assertEqual(sorted(released), [1, 2])
