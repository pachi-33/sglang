import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.moe_trace.codec import dequantize_packed_int4
from sglang.srt.moe_trace.config import MoeTraceConfig
from sglang.srt.moe_trace.recorder import MoeTraceRecorder
from sglang.srt.moe_trace.types import MoeTraceSite
from sglang.srt.moe_trace.writer import safe_request_id


class _Mode:
    def __init__(self, decode: bool):
        self.decode = decode

    def is_decode(self):
        return self.decode


def _batch(*, decode=True, request_ids=("r0", "r1")):
    rows = len(request_ids)
    return SimpleNamespace(
        forward_mode=_Mode(decode),
        rids=list(request_ids),
        batch_size=rows,
        input_ids=torch.arange(10, 10 + rows, dtype=torch.int64),
        positions=torch.arange(20, 20 + rows, dtype=torch.int64),
    )


def _config(root: str, *, routes: bool, activations: bool, max_tokens: int = 0):
    return MoeTraceConfig(
        output_dir=Path(root),
        expert_routes=routes,
        router_inputs=activations,
        max_decode_tokens=max_tokens,
        activation_group_size=2,
        queue_depth=2,
        overflow_policy="block",
    )


class TestMoeTraceRecorder(unittest.TestCase):
    def _recorder(self, root, *, routes, activations, max_tokens=0):
        site = MoeTraceSite(0, "layers.0.topk", 0, "topk", 4, 8, 2)
        with patch(
            "sglang.srt.moe_trace.recorder.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        ):
            return MoeTraceRecorder(
                _config(
                    root,
                    routes=routes,
                    activations=activations,
                    max_tokens=max_tokens,
                ),
                [site],
                max_rows=4,
                device="cpu",
            )

    def test_feature_matrix_and_decode_only(self):
        for routes, activations in ((True, False), (False, True), (True, True)):
            with self.subTest(routes=routes, activations=activations):
                with tempfile.TemporaryDirectory() as root:
                    recorder = self._recorder(
                        root, routes=routes, activations=activations
                    )
                    module = nn.Module()
                    module._moe_trace_site_id = 0
                    hidden = torch.tensor(
                        [[-7.0, 3.5, 0.0, 1.0], [2.0, -2.0, 4.0, -4.0]]
                    )
                    ids = torch.tensor([[3, 1], [7, 2]], dtype=torch.int32)
                    weights = torch.tensor(
                        [[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32
                    )

                    recorder.begin_forward(True)
                    with recorder.capture_scope():
                        recorder.capture_router_input(module, hidden)
                        recorder.capture_route(module, ids, weights)
                    payload = recorder.end_forward(_batch())

                    self.assertIsNotNone(payload)
                    self.assertTrue(torch.all(payload.site_valid))
                    self.assertEqual(payload.request_ids, ["r0", "r1"])
                    self.assertEqual(payload.input_token_ids.tolist(), [10, 11])
                    if activations:
                        restored = dequantize_packed_int4(
                            payload.activation_q,
                            payload.activation_scales,
                            hidden_size=4,
                            group_size=2,
                            dtype=torch.float32,
                        )
                        self.assertEqual(tuple(payload.activation_q.shape), (2, 2))
                        self.assertTrue(torch.allclose(restored, hidden, atol=0.6))
                    else:
                        self.assertIsNone(payload.activation_q)
                        self.assertIsNone(payload.activation_scales)
                    if routes:
                        self.assertTrue(torch.equal(payload.expert_ids, ids))
                        self.assertTrue(torch.equal(payload.expert_weights, weights))
                    else:
                        self.assertIsNone(payload.expert_ids)
                        self.assertIsNone(payload.expert_weights)

                    self.assertIsNone(recorder.end_forward(_batch(decode=False)))
                    recorder.close()

    def test_max_decode_tokens_is_per_request(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(
                root, routes=True, activations=False, max_tokens=1
            )
            module = nn.Module()
            module._moe_trace_site_id = 0
            with recorder.capture_scope():
                recorder.capture_route(
                    module,
                    torch.ones((2, 2), dtype=torch.int32),
                    torch.ones((2, 2), dtype=torch.float32),
                )
            self.assertIsNotNone(recorder.end_forward(_batch()))
            self.assertIsNone(recorder.end_forward(_batch()))
            payload = recorder.end_forward(_batch(request_ids=("r0", "new")))
            self.assertEqual(payload.request_ids, ["new"])
            self.assertEqual(payload.input_token_ids.tolist(), [11])
            recorder.close()

    def test_disabled_config_and_validation(self):
        disabled = SimpleNamespace(
            moe_trace_output_dir=None,
            moe_trace_expert_routes=False,
            moe_trace_router_inputs=False,
            moe_trace_max_decode_tokens=0,
            moe_trace_activation_group_size=128,
            moe_trace_queue_depth=2,
            moe_trace_overflow_policy="block",
        )
        self.assertIsNone(MoeTraceConfig.from_observability(disabled))
        disabled.moe_trace_router_inputs = True
        with self.assertRaisesRegex(ValueError, "output-dir"):
            MoeTraceConfig.from_observability(disabled)

    def test_overlap_overshoot_is_dropped_before_finalization(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root, routes=True, activations=False)
            module = nn.Module()
            module._moe_trace_site_id = 0

            def capture_batch():
                recorder.begin_forward(True)
                with recorder.capture_scope():
                    recorder.capture_route(
                        module,
                        torch.ones((2, 2), dtype=torch.int32),
                        torch.ones((2, 2), dtype=torch.float32),
                    )
                return recorder.end_forward(_batch())

            recorder.submit(capture_batch())
            recorder.finalize_request("r0", defer=True)
            recorder.submit(capture_batch())
            recorder.close()

            def manifest(request_id):
                path = Path(root) / safe_request_id(request_id) / "manifest.json"
                return json.loads(path.read_text())

            self.assertEqual(manifest("r0")["total_rows"], 1)
            self.assertEqual(manifest("r0")["status"], "complete")
            self.assertEqual(manifest("r1")["total_rows"], 2)
            self.assertEqual(manifest("r1")["status"], "truncated")


if __name__ == "__main__":
    unittest.main()
