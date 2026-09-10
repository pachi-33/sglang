"""CPU-only contract tests for opt-in decode timing."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.qwen3_5 import decode_timing
from sglang.srt.layers.qwen3_5 import moe as moe_module


class _FakeEvent:
    clock = 0.0

    def __init__(self, *, enable_timing):
        if enable_timing is not True:
            raise AssertionError("timing events must enable CUDA timing")
        self.value = None

    def record(self):
        self.value = float(self.clock)

    def elapsed_time(self, other):
        if self.value is None or other.value is None:
            raise AssertionError("elapsed_time used an unrecorded event")
        return other.value - self.value


class TestDecodeTimingCapture(unittest.TestCase):
    def setUp(self):
        self.event_patch = mock.patch.object(
            decode_timing.torch.cuda, "Event", side_effect=_FakeEvent
        )
        self.event_patch.start()
        self.addCleanup(self.event_patch.stop)

    @staticmethod
    def _record_complete_step(capture, start=10.0):
        _FakeEvent.clock = start
        capture.begin_step()
        value = start + 1.0
        for layer_id in capture.layer_ids:
            for boundary in decode_timing.DECODE_TIMING_BOUNDARIES:
                _FakeEvent.clock = value
                capture.record(layer_id, boundary)
                value += 1.0
        return capture.finish_step()

    def test_complete_step_is_monotonic_and_reuses_events(self):
        capture = decode_timing.DecodeTimingCapture((17, 18), device="cuda:0")
        first = self._record_complete_step(capture)
        self.assertEqual(first["format"], decode_timing.DECODE_TIMING_FORMAT)
        self.assertEqual(first["layer_ids"], [17, 18])
        self.assertEqual(first["elapsed_ms"][0], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(first["elapsed_ms"][1], [6.0, 7.0, 8.0, 9.0, 10.0])
        self.assertIs(capture.last_result, first)

        second = self._record_complete_step(capture, start=100.0)
        self.assertEqual(second["elapsed_ms"], first["elapsed_ms"])

    def test_event_order_and_incomplete_steps_are_rejected(self):
        capture = decode_timing.DecodeTimingCapture((17,), device="cuda")
        _FakeEvent.clock = 0.0
        capture.begin_step()
        with self.assertRaisesRegex(RuntimeError, "event order"):
            capture.record(17, "router_start")
        capture.abort_step()
        self.assertIsNone(capture.last_result)

        capture.begin_step()
        capture.record(17, "layer_start")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            capture.finish_step()

    def test_invalid_layer_ids_and_cpu_device_are_rejected(self):
        for ids in ((), (18, 17), (17, 17)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                decode_timing.DecodeTimingCapture(ids, device="cuda")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            decode_timing.DecodeTimingCapture((17,), device="cpu")


class TestMoeTimingHook(unittest.TestCase):
    def test_resident_path_marks_router_and_routed_boundaries_in_order(self):
        x = torch.zeros((1, 2), dtype=torch.float16)
        ids = torch.zeros((1, 8), dtype=torch.int32)
        probabilities = torch.full((1, 8), 0.125, dtype=torch.float32)
        output = torch.ones_like(x)
        weights = moe_module.MoeWeights(
            router=SimpleNamespace(),
            gate_up=SimpleNamespace(kind="nvfp4"),
            down=SimpleNamespace(),
        )
        observed = []
        with mock.patch.object(
            moe_module, "_fp16_linear", return_value=torch.empty((1, 256))
        ), mock.patch.object(
            moe_module, "route_topk", return_value=(ids, probabilities)
        ), mock.patch.object(
            moe_module, "execute_experts", return_value=output
        ) as execute:
            actual = moe_module.fused_moe(x, weights, timing_hook=observed.append)

        self.assertIs(actual, output)
        self.assertEqual(
            observed, ["router_start", "router_ready", "routed_expert_start"]
        )
        execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
