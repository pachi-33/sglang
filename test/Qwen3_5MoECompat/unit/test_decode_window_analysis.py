"""CPU-only tests for resident decode window post-processing."""

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.expert_prefetch import analyze_decode_window as analysis
from experiments.expert_prefetch import profile_resident_decode_window as profile


def _timing_layer(layer_id, start):
    return {
        "layer_id": layer_id,
        "layer_start_ms": float(start),
        "router_start_ms": float(start + 1),
        "router_ready_ms": float(start + 2),
        "routed_expert_start_ms": float(start + 3),
        "layer_end_ms": float(start + 4),
    }


class TestDecodeWindowAnalysis(unittest.TestCase):
    def test_flatten_and_capacity_analysis_preserve_axes(self):
        steps = [
            {
                "step_id": step + 1,
                "prefix_len": 1024 + step,
                "role": "back",
                "layers": [
                    _timing_layer(17, 0),
                    _timing_layer(18, 5),
                    _timing_layer(19, 10),
                ],
            }
            for step in range(2)
        ]
        flattened = profile._flatten_timings(steps, (17, 18, 19))
        self.assertEqual(len(flattened), 6)
        self.assertEqual(flattened[0]["layer_total_ms"], 4.0)
        self.assertEqual(flattened[-1]["decode_index"], 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decode_jsonl = root / "decode.jsonl"
            decode_jsonl.write_text(
                "".join(json.dumps(row) + "\n" for row in flattened),
                encoding="utf-8",
            )
            pp_summary = root / "pp.json"
            pp_summary.write_text(
                json.dumps(
                    {
                        "measurement_decode_steps": 2,
                        "back_layer_ids": [17, 18, 19],
                        "token_checks": {
                            "first_oracle_tokens_match_exp0001": True,
                            "timed_equals_control": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            h2d_npz = root / "h2d.npz"
            np.savez_compressed(
                h2d_npz,
                format=np.asarray("SGLANG-QWEN35-PINNED-H2D-BENCHMARK-v1"),
                candidate_counts=np.asarray([1, 2], dtype=np.int32),
                transfer_bytes=np.asarray([100, 200], dtype=np.int64),
                latency_ms=np.asarray([[1.0, 1.0], [7.0, 7.0]], dtype=np.float64),
                payload_size=np.asarray(100, dtype=np.int64),
                warmup=np.asarray(0, dtype=np.int32),
                samples=np.asarray(2, dtype=np.int32),
                source_payload_sha256=np.asarray(["a", "b"], dtype="<U1"),
            )
            args = argparse.Namespace(
                decode_jsonl=decode_jsonl,
                pp_summary=pp_summary,
                h2d_npz=h2d_npz,
                layer_summary=root / "layers.csv",
                window_summary=root / "windows.csv",
                h2d_summary=root / "h2d.csv",
                capacity_summary=root / "capacity.csv",
                experiment_summary=root / "summary.json",
                layer_start=17,
                layer_end=20,
                expected_steps=2,
                expected_h2d_samples=2,
            )

            summary = analysis.analyze(args)

            self.assertEqual(summary["layer_samples"], 6)
            self.assertEqual(summary["adjacent_window_samples"], 4)
            self.assertEqual(summary["offload_relevant_window_samples"], 4)
            self.assertEqual(summary["overall_max_k_95"], 1)
            with args.window_summary.open(newline="", encoding="utf-8") as file:
                windows = list(csv.DictReader(file))
            self.assertEqual(len(windows), 2)
            self.assertEqual(
                float(windows[0]["route_to_next_routed_expert_start_p50_ms"]),
                6.0,
            )
            with args.capacity_summary.open(newline="", encoding="utf-8") as file:
                capacity = list(csv.DictReader(file))
            self.assertEqual(len(capacity), 4)
            self.assertEqual(capacity[0]["transition_max_k_95"], "1")


if __name__ == "__main__":
    unittest.main()
