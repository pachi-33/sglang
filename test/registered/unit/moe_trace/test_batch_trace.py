import ast
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.moe_trace.batch_writer import MoeTraceBatchWriter
from sglang.srt.moe_trace.config import MoeTraceConfig
from sglang.srt.moe_trace.loader import MoeTraceBatchLoader
from sglang.srt.moe_trace.recorder import MoeTraceRecorder, build_site_layouts
from sglang.srt.moe_trace.types import MoeTraceBatchOutput, MoeTraceSite, TraceRowLayout


class _Mode:
    def __init__(self, phase):
        self.phase = phase

    def is_decode(self):
        return self.phase == "decode"

    def is_target_verify(self):
        return self.phase == "target_verify"


def _batch(request_ids=("A", "B"), width=3, padded_rows=8, phase="target_verify"):
    return SimpleNamespace(
        forward_mode=_Mode(phase),
        rids=list(request_ids),
        batch_size=padded_rows // width,
        input_ids=torch.arange(padded_rows),
        positions=torch.tensor([10 + i % width for i in range(padded_rows)]),
        spec_info=SimpleNamespace(num_tokens_per_req=width),
    )


def _sites():
    return [MoeTraceSite(0, "moe.topk", 0, "topk", 4, 32, 2)]


def _payload(forward_id=0, request_ids=("A", "B"), counts=(3, 2)):
    layout = TraceRowLayout.from_counts(
        forward_id, "target_verify", request_ids, counts
    )
    rows = sum(counts)
    return MoeTraceBatchOutput(
        request_ids=layout.expanded_request_ids(),
        input_token_ids=torch.arange(rows),
        positions=torch.arange(rows),
        site_valid=torch.ones(rows, 1, dtype=torch.bool),
        expert_ids=torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2),
        row_layout=layout,
    )


class TestMoeTraceMultirowRecorder(unittest.TestCase):
    def _recorder(self, root, limit=0):
        config = MoeTraceConfig(
            Path(root), True, False, limit, 128, 2, "block", ids_only=True
        )
        with patch(
            "sglang.srt.moe_trace.recorder.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        ):
            return MoeTraceRecorder(config, _sites(), max_rows=8, device="cpu")

    def _capture(self, recorder, ids=None):
        module = nn.Module()
        module._moe_trace_site_id = 0
        ids = torch.arange(16, dtype=torch.int32).reshape(8, 2) if ids is None else ids
        recorder.begin_forward(True)
        with recorder.capture_scope():
            recorder.capture_route(
                module, ids, torch.ones_like(ids, dtype=torch.float32)
            )

    def test_concurrent_verify_rows_exclude_graph_padding_and_weights(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root)
            try:
                batch = _batch()
                context = recorder.prepare_forward(batch)
                self._capture(recorder)
                payload = recorder.end_forward(batch, context=context)
                self.assertEqual(payload.request_ids, ["A"] * 3 + ["B"] * 3)
                self.assertEqual(payload.expert_ids.shape, (6, 2))
                self.assertEqual(payload.row_layout.row_splits, (0, 3, 6))
                self.assertIsNone(recorder.expert_weights)
                self.assertIsNone(recorder.activation_q)
                self.assertIsNone(payload.expert_weights)
                self.assertEqual(payload.positions.tolist(), [10, 11, 12, 10, 11, 12])
            finally:
                recorder.close()

    def test_nonuniform_layout_gathers_holes_and_freezes_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root)
            try:
                batch = _batch()
                layout = TraceRowLayout.from_counts(
                    7, "target_verify", ("A", "B"), (3, 2), (0, 1, 2, 4, 5)
                )
                context = recorder.prepare_rows(
                    layout, batch.input_ids, batch.positions
                )
                batch.input_ids.fill_(-100)
                self._capture(recorder)
                payload = recorder.end_forward(batch, context=context)
                self.assertEqual(payload.input_token_ids.tolist(), [0, 1, 2, 4, 5])
                self.assertEqual(payload.expert_ids[:, 0].tolist(), [0, 2, 4, 8, 10])
                self.assertEqual(payload.row_layout.candidate_indices, (0, 1, 2, 0, 1))
            finally:
                recorder.close()

    def test_128_concurrent_requests_with_eight_candidates_each(self):
        with tempfile.TemporaryDirectory() as root:
            config = MoeTraceConfig(
                Path(root), True, False, 0, 128, 2, "block", ids_only=True
            )
            with patch(
                "sglang.srt.moe_trace.recorder.get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ):
                recorder = MoeTraceRecorder(
                    config, _sites(), max_rows=2048, device="cpu"
                )
            try:
                batch = _batch(
                    tuple(f"r{index}" for index in range(128)),
                    width=8,
                    padded_rows=2048,
                )
                context = recorder.prepare_forward(batch)
                module = SimpleNamespace(_moe_trace_site_id=0)
                ids = torch.arange(4096, dtype=torch.int32).reshape(2048, 2) % 32
                recorder.begin_forward(True)
                with recorder.capture_scope():
                    recorder.capture_route(
                        module, ids, torch.ones_like(ids, dtype=torch.float32)
                    )
                payload = recorder.end_forward(batch, context=context)
                self.assertEqual(payload.expert_ids.shape, (1024, 2))
                self.assertEqual(payload.request_ids[-8:], ["r127"] * 8)
                recorder.submit(payload)
            finally:
                recorder.close()
            loader = MoeTraceBatchLoader(root)
            self.assertEqual(loader.manifest()["total_rows"], 1024)
            self.assertTrue(
                torch.equal(
                    loader.load_trace("r127")["site.0.expert_ids"], ids[1016:1024]
                )
            )

    def test_snapshot_reorder_and_finished_request_overlap_are_retained(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root)
            batch = _batch()
            first_context = recorder.prepare_forward(batch)
            self._capture(recorder)
            first = recorder.end_forward(batch, context=first_context)
            expected = first.expert_ids.clone()
            recorder.submit(first)
            recorder.finalize_request("A", defer=True)
            self._capture(recorder, torch.full((8, 2), 17, dtype=torch.int32))
            second_batch = _batch(("B", "A"))
            second_context = recorder.prepare_forward(second_batch)
            second = recorder.end_forward(second_batch, context=second_context)
            self.assertTrue(torch.equal(first.expert_ids, expected))
            self.assertEqual(second.request_ids, ["B"] * 3 + ["A"] * 3)
            recorder.submit(second)
            recorder.close()
            loader = MoeTraceBatchLoader(root)
            trace = loader.load_trace("A")
            self.assertEqual(trace["forward_ids"].tolist(), [0, 0, 0, 1, 1, 1])
            self.assertEqual(trace["site.0.expert_ids"][-3:].tolist(), [[17, 17]] * 3)
            self.assertEqual(loader.manifest()["requests"]["A"]["status"], "complete")

    def test_limit_counts_candidate_rows_per_request(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root, limit=2)
            try:
                self._capture(recorder)
                payload = recorder.end_forward(_batch())
                self.assertEqual(payload.request_ids, ["A", "A", "B", "B"])
                self.assertEqual(payload.row_layout.row_splits, (0, 2, 4))
                self.assertEqual(payload.row_layout.capture_indices, (0, 1, 3, 4))
                self.assertIsNone(recorder.end_forward(_batch()))
            finally:
                recorder.close()

    def test_many_overlapping_batches_with_arrival_departure_and_reordering(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root)
            expected = {}
            for forward_id in range(40):
                requests = ("AB", "CA", "DC", "BD")[forward_id % 4]
                width = 1 + forward_id % 3
                batch = _batch(tuple(requests), width=width)
                context = recorder.prepare_forward(batch)
                self._capture(
                    recorder, torch.full((8, 2), forward_id % 32, dtype=torch.int32)
                )
                payload = recorder.end_forward(batch, context=context)
                recorder.submit(payload)
                for request_id in requests:
                    expected.setdefault(request_id, []).extend(
                        [forward_id % 32] * width
                    )
            recorder.close()
            loader = MoeTraceBatchLoader(root)
            self.assertEqual(len(list(loader.iter_forwards())), 40)
            for request_id, values in expected.items():
                self.assertEqual(
                    loader.load_trace(request_id)["site.0.expert_ids"][:, 0].tolist(),
                    values,
                )

    def test_invalid_layouts_fail_before_capture(self):
        for counts, indices in (((2,), (0, 0)), ((2,), (0,))):
            with (
                self.subTest(counts=counts, indices=indices),
                self.assertRaises(ValueError),
            ):
                TraceRowLayout.from_counts(0, "decode", ("A",), counts, indices)
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root)
            try:
                with self.assertRaisesRegex(ValueError, "capacity"):
                    recorder.prepare_forward(_batch(width=5, padded_rows=10))
                with self.assertRaisesRegex(ValueError, "decode or target verify"):
                    recorder.prepare_forward(_batch(phase="draft"))
            finally:
                recorder.close()


class TestMoeTraceBatchWriter(unittest.TestCase):
    def _writer(self, root, **kwargs):
        return MoeTraceBatchWriter(
            root, build_site_layouts(_sites(), 128), fsync=False, **kwargs
        )

    def test_round_trip_aggregates_forwards_and_masks_unvisited_sites(self):
        with tempfile.TemporaryDirectory() as root:
            writer = self._writer(root, chunk_forwards=2, flush_interval=60)
            first, second = _payload(), _payload(1, ("B", "C"), (2, 1))
            first.site_valid[1] = False
            released = Mock()
            first.release_callback = released
            writer.submit(first)
            writer.submit(second)
            writer.close()
            loader = MoeTraceBatchLoader(root)
            self.assertEqual(len(loader.manifest()["chunks"]), 1)
            frames = list(loader.iter_forwards())
            self.assertEqual([frame[0]["forward_id"] for frame in frames], [0, 1])
            self.assertEqual(
                loader.load_trace("B")["forward_ids"].tolist(), [0, 0, 1, 1]
            )
            self.assertEqual(
                loader.load_trace("A")["site.0.expert_ids"][1].tolist(), [-1, -1]
            )
            self.assertNotIn("site.0.expert_weights", frames[0][1])
            released.assert_called_once_with()

    def test_drop_queue_releases_only_owned_payloads_once(self):
        with tempfile.TemporaryDirectory() as root:
            writer = self._writer(
                root, queue_depth=1, overflow="drop", chunk_forwards=1
            )
            started, resume = threading.Event(), threading.Event()
            original = writer._write_frames

            def blocked(payloads):
                started.set()
                if not resume.wait(5):
                    raise RuntimeError("test writer was not resumed")
                original(payloads)

            writer._write_frames = blocked
            payloads = [_payload(index) for index in range(3)]
            for payload in payloads:
                payload.release_callback = Mock()
            try:
                self.assertTrue(writer.submit(payloads[0]))
                self.assertTrue(started.wait(5))
                self.assertTrue(writer.submit(payloads[1]))
                self.assertFalse(writer.submit(payloads[2]))
            finally:
                resume.set()
                writer.close()
            for payload in payloads:
                self.assertIsNone(payload.release_callback)
            self.assertEqual(writer.stats["dropped_rows"], 5)

    def test_out_of_order_forward_surfaces_failure_and_releases_payload(self):
        with tempfile.TemporaryDirectory() as root:
            writer = self._writer(root)
            first, second = _payload(2), _payload(1)
            released = Mock()
            second.release_callback = released
            writer.submit(first)
            writer.submit(second)
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                writer.close()
            self.assertEqual(
                json.loads((Path(root) / "manifest.json").read_text())["status"],
                "failed",
            )
            released.assert_called_once_with()

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            writer = self._writer(root)
            writer.close()
            before = (Path(root) / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self._writer(root)
            self.assertEqual((Path(root) / "manifest.json").read_bytes(), before)

    def test_metadata_hash_and_row_ranges_are_validated(self):
        with tempfile.TemporaryDirectory() as root:
            writer = self._writer(root)
            writer.submit(_payload())
            writer.close()
            loader = MoeTraceBatchLoader(root)
            metadata_path = Path(root) / loader.manifest()["chunks"][0]["metadata_path"]
            metadata = json.loads(metadata_path.read_text())
            metadata["forwards"][0]["row_start"] = 1
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                list(loader.iter_chunks())
            with self.assertRaisesRegex(ValueError, "row range"):
                list(loader.iter_chunks(verify=False))


class TestMoeTraceTargetIntegration(unittest.TestCase):
    def test_eager_verify_capacity_includes_maximum_adaptive_width(self):
        root = Path(__file__).parents[4]
        tree = ast.parse(
            (root / "python/sglang/srt/model_executor/model_runner.py").read_text()
        )
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_moe_trace_recorder"
        )
        factory = Mock()
        namespace = {
            "get_parallel": lambda: SimpleNamespace(tp_rank=0),
            "max_speculative_num_draft_tokens": lambda: 12,
            "MoeTraceRecorder": factory,
            "set_global_moe_trace_recorder": Mock(),
        }
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]), "<trace capacity>", "exec"
            ),
            namespace,
        )
        runner = SimpleNamespace(
            moe_trace_config=SimpleNamespace(ids_only=True),
            max_running_requests=128,
            max_decode_logits_rows=lambda: 64,
            decode_num_tokens_per_req=Mock(return_value=12),
            moe_trace_sites=[],
            device="npu",
        )
        namespace["init_moe_trace_recorder"](runner)
        self.assertEqual(factory.call_args.kwargs["max_rows"], 1536)
        runner.decode_num_tokens_per_req.assert_called_once_with(num_draft_tokens=12)

    def test_runtime_capture_guard_excludes_draft_prefill_and_idle(self):
        root = Path(__file__).parents[4]
        tree = ast.parse(
            (root / "python/sglang/srt/model_executor/model_runner.py").read_text()
        )
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "trace_this_forward"
                for target in node.targets
            )
        )
        expression = compile(
            ast.Expression(assignment.value), "<target trace guard>", "eval"
        )
        for draft in (False, True):
            for ids_only in (False, True):
                for phase in ("decode", "target_verify", "prefill", "idle", "draft"):
                    namespace = {
                        "self": SimpleNamespace(is_draft_worker=draft),
                        "trace_recorder": SimpleNamespace(
                            config=SimpleNamespace(ids_only=ids_only)
                        ),
                        "forward_batch": _batch(phase=phase),
                    }
                    self.assertEqual(
                        eval(expression, namespace),
                        not draft
                        and (
                            phase == "decode" or ids_only and phase == "target_verify"
                        ),
                    )

    def test_eagle_verify_result_keeps_target_trace_payload(self):
        root = Path(__file__).parents[4]
        tree = ast.parse(
            (root / "python/sglang/srt/speculative/eagle_worker_common.py").read_text()
        )
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_eagle_verify"
        )
        returned = next(
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "GenerationBatchResult"
        )
        sentinel = object()
        namespace = {
            node.id: None for node in ast.walk(returned) if isinstance(node, ast.Name)
        }
        namespace.update(
            GenerationBatchResult=SimpleNamespace,
            forward_batch_output=SimpleNamespace(
                routed_experts_output=None,
                indexer_topk_output=None,
                moe_trace_output=sentinel,
            ),
        )
        result = eval(
            compile(ast.Expression(returned), "<eagle result>", "eval"), namespace
        )
        self.assertIs(result.moe_trace_output, sentinel)
