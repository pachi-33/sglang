"""CPU-only contract tests for per-request logical expert traces."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from sglang.srt.layers.qwen3_5 import expert_trace


class TestExpertTrace(unittest.TestCase):
    @staticmethod
    def _capture_all(step, *, tokens=1, offset=0):
        expected = np.empty((40, 8), dtype=np.uint8)
        for layer_id in range(40):
            rows = torch.empty((tokens, 8), dtype=torch.int32)
            for token in range(tokens):
                rows[token] = torch.tensor(
                    [(offset + layer_id + token + rank) % 256 for rank in range(8)],
                    dtype=torch.int32,
                )
            step.capture(layer_id, rows)
            expected[layer_id] = rows[-1].numpy().astype(np.uint8)
        return expected

    def test_output_attribution_arrays_and_atomic_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            config = expert_trace.ExpertTraceConfig(directory)
            session = expert_trace.ExpertTraceSession(
                config,
                request_id="cmpl-test",
                max_rows=3,
                prompt_tokens=5,
                device="cpu",
                identity={"model": {"config_sha256": "a" * 64}},
            )
            prefill = session.begin_step(
                phase=expert_trace.PHASE_PREFILL_LAST,
                model_input_token_id=14,
                model_input_position=4,
            )
            expected_prefill = self._capture_all(prefill, tokens=5)
            session.commit_step(prefill, 31)

            decode = session.begin_step(
                phase=expert_trace.PHASE_DECODE,
                model_input_token_id=31,
                model_input_position=5,
            )
            expected_decode = self._capture_all(decode, offset=100)
            session.commit_step(decode, 32)
            metadata = session.finalize(status="ok", stopped_on_eos=False, error=None)

            self.assertEqual(session.state, "COMMITTED")
            self.assertTrue(session.metadata_path.is_file())
            self.assertTrue(session.data_path.is_file())
            self.assertFalse(Path(str(session.metadata_path) + ".partial").exists())
            self.assertFalse(Path(str(session.data_path) + ".partial").exists())

            with np.load(session.data_path, allow_pickle=False) as arrays:
                self.assertEqual(
                    arrays.files,
                    [
                        "expert_ids",
                        "sampled_token_ids",
                        "model_input_token_ids",
                        "model_input_positions",
                        "phase",
                    ],
                )
                self.assertEqual(arrays["expert_ids"].dtype, np.uint8)
                self.assertEqual(arrays["expert_ids"].shape, (2, 40, 8))
                np.testing.assert_array_equal(arrays["expert_ids"][0], expected_prefill)
                np.testing.assert_array_equal(arrays["expert_ids"][1], expected_decode)
                np.testing.assert_array_equal(arrays["sampled_token_ids"], [31, 32])
                np.testing.assert_array_equal(arrays["model_input_token_ids"], [14, 31])
                np.testing.assert_array_equal(arrays["model_input_positions"], [4, 5])
                np.testing.assert_array_equal(arrays["phase"], [0, 1])

            parsed = json.loads(session.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed, metadata)
            self.assertEqual(parsed["format"], expert_trace.FORMAT_ID)
            self.assertEqual(parsed["status"], "ok")
            self.assertEqual(parsed["request"]["completion_tokens"], 2)
            self.assertEqual(parsed["routing"]["layer_ids"], list(range(40)))
            digest = hashlib.sha256(session.data_path.read_bytes()).hexdigest()
            self.assertEqual(parsed["artifact"]["sha256"], digest)

    def test_failed_request_excludes_uncommitted_pending_row(self):
        with tempfile.TemporaryDirectory() as directory:
            session = expert_trace.ExpertTraceSession(
                expert_trace.ExpertTraceConfig(directory),
                request_id="failed-test",
                max_rows=2,
                prompt_tokens=1,
                device="cpu",
                identity={},
            )
            pending = session.begin_step(
                phase=expert_trace.PHASE_PREFILL_LAST,
                model_input_token_id=7,
                model_input_position=0,
            )
            pending.capture(0, torch.arange(8, dtype=torch.int32).view(1, 8))
            failure = RuntimeError("model failed")
            metadata = session.finalize(
                status="failed", stopped_on_eos=False, error=failure
            )
            with np.load(session.data_path, allow_pickle=False) as arrays:
                self.assertEqual(arrays["expert_ids"].shape, (0, 40, 8))
                self.assertEqual(arrays["sampled_token_ids"].shape, (0,))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["request"]["completed_rows"], 0)
            self.assertEqual(metadata["error"]["type"], "RuntimeError")

    def test_missing_layer_and_out_of_range_id_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            session = expert_trace.ExpertTraceSession(
                expert_trace.ExpertTraceConfig(directory),
                request_id="invalid-test",
                max_rows=1,
                prompt_tokens=1,
                device="cpu",
                identity={},
            )
            step = session.begin_step(
                phase=expert_trace.PHASE_PREFILL_LAST,
                model_input_token_id=1,
                model_input_position=0,
            )
            step.capture(0, torch.arange(8, dtype=torch.int32).view(1, 8))
            with self.assertRaisesRegex(RuntimeError, "missing layers"):
                session.commit_step(step, 2)

            for layer_id in range(1, 40):
                values = torch.arange(8, dtype=torch.int32).view(1, 8)
                if layer_id == 39:
                    values[0, 0] = 256
                step.capture(layer_id, values)
            session.commit_step(step, 2)
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                session.finalize(status="ok", stopped_on_eos=False, error=None)
            self.assertEqual(session.state, "FAILED")

    def test_serialization_failure_removes_partial_and_uncommitted_data(self):
        with tempfile.TemporaryDirectory() as directory:
            session = expert_trace.ExpertTraceSession(
                expert_trace.ExpertTraceConfig(directory),
                request_id="io-test",
                max_rows=1,
                prompt_tokens=1,
                device="cpu",
                identity={},
            )
            step = session.begin_step(
                phase=expert_trace.PHASE_PREFILL_LAST,
                model_input_token_id=1,
                model_input_position=0,
            )
            self._capture_all(step)
            session.commit_step(step, 2)
            with mock.patch.object(
                expert_trace.np,
                "savez_compressed",
                side_effect=OSError("disk full"),
            ), self.assertRaisesRegex(OSError, "disk full"):
                session.finalize(status="ok", stopped_on_eos=False, error=None)
            self.assertEqual(session.state, "FAILED")
            self.assertFalse(session.metadata_path.exists())
            self.assertFalse(session.data_path.exists())
            self.assertFalse(Path(str(session.data_path) + ".partial").exists())

    def test_directory_fsync_failure_removes_published_commit_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            session = expert_trace.ExpertTraceSession(
                expert_trace.ExpertTraceConfig(directory),
                request_id="fsync-test",
                max_rows=1,
                prompt_tokens=1,
                device="cpu",
                identity={},
            )
            step = session.begin_step(
                phase=expert_trace.PHASE_PREFILL_LAST,
                model_input_token_id=1,
                model_input_position=0,
            )
            self._capture_all(step)
            session.commit_step(step, 2)
            with mock.patch.object(
                expert_trace,
                "_fsync_directory",
                side_effect=OSError("fsync failed"),
            ), self.assertRaisesRegex(OSError, "fsync failed"):
                session.finalize(status="ok", stopped_on_eos=False, error=None)
            self.assertEqual(session.state, "FAILED")
            self.assertFalse(session.metadata_path.exists())
            self.assertFalse(session.data_path.exists())


if __name__ == "__main__":
    unittest.main()
