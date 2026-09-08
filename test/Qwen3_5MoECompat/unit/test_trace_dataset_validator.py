import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = REPO_ROOT / "experiments/expert_prefetch/validate_trace_dataset.py"
SPEC = importlib.util.spec_from_file_location("validate_trace_dataset", VALIDATOR_PATH)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTraceDatasetValidator(unittest.TestCase):
    def test_validates_and_indexes_complete_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_dir = root / "traces"
            trace_dir.mkdir()
            results = root / "results"
            results.mkdir()
            trace_id = "cmpl-test"
            npz_path = trace_dir / f"{trace_id}.trace.npz"
            arrays = {
                "expert_ids": np.arange(3 * 40 * 8, dtype=np.uint8).reshape(3, 40, 8),
                "sampled_token_ids": np.array([7, 8, 9], dtype=np.int32),
                "model_input_token_ids": np.array([6, 7, 8], dtype=np.int32),
                "model_input_positions": np.array([3, 4, 5], dtype=np.int32),
                "phase": np.array([0, 1, 1], dtype=np.uint8),
            }
            np.savez_compressed(npz_path, **arrays)
            metadata = {
                "format": validator.TRACE_FORMAT,
                "trace_id": trace_id,
                "status": "ok",
                "started_at": "2026-09-08T00:00:00Z",
                "finished_at": "2026-09-08T00:00:01Z",
                "request": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "completed_rows": 3,
                },
                "identity": {
                    "model": {"config_sha256": "config"},
                    "expert_pack": {"pack_sha256": "pack"},
                    "device": {"uuid": "GPU-test"},
                },
                "artifact": {
                    "file": npz_path.name,
                    "size": npz_path.stat().st_size,
                    "sha256": _sha256(npz_path),
                },
            }
            json_path = trace_dir / f"{trace_id}.trace.json"
            json_path.write_text(json.dumps(metadata), encoding="utf-8")
            bench_path = root / "bench.jsonl"
            bench_path.write_text(
                json.dumps(
                    {
                        "expert_trace": True,
                        "benchmark_duration": 1.0,
                        "output_token_throughput": 3.0,
                        "trace_requests": [
                            {
                                "request_index": 0,
                                "trace_id": trace_id,
                                "success": True,
                                "prompt_kind": "token_ids_int32_le",
                                "prompt_sha256": "prompt",
                                "prompt_tokens": 4,
                                "reported_prompt_tokens": 4,
                                "completion_tokens": 3,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = results / "dataset_manifest.jsonl"
            summary_path = results / "dataset_summary.json"

            summary = validator.validate_dataset(
                bench_jsonl=bench_path,
                trace_dir=trace_dir,
                output_manifest=manifest_path,
                output_summary=summary_path,
                expected_count=1,
                prompt_tokens=4,
                completion_tokens=3,
                gpu_uuid="GPU-test",
            )

            self.assertEqual(summary["logical_dataset_shape"], [1, 3, 40, 8])
            self.assertEqual(summary["total_expert_selections"], 960)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(summary_path.is_file())


if __name__ == "__main__":
    unittest.main()
