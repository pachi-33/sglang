import os
import unittest
from pathlib import Path

from sglang.srt.layers.qwen3_5.checkpoint import Qwen35Checkpoint
from sglang.srt.layers.qwen3_5.manifest import EXPECTED_HEADERS

MODEL_DIR = Path(
    os.environ.get(
        "QWEN35_MODEL_DIR",
        "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
    )
)


class TestCheckpointManifest(unittest.TestCase):
    @unittest.skipUnless(MODEL_DIR.is_dir(), "Qwen3.5 checkpoint unavailable")
    def test_quantized_matrix_manifest(self):
        audit = Qwen35Checkpoint(MODEL_DIR).audit()
        self.assertEqual(
            (audit.layers, audit.fp8_matrices, audit.nvfp4_matrices), (40, 130, 29184)
        )
        self.assertEqual(audit.fp16_expert_matrices, 1536)


class TestWeightContract(unittest.TestCase):
    def test_fp16_disallows_quantization_metadata(self):
        import torch

        from sglang.srt.layers.qwen3_5.weights import Weight

        with self.assertRaises(ValueError):
            Weight(
                "fp16", torch.zeros((2, 2), dtype=torch.float16), (2, 2), torch.ones(1)
            )


class TestExactHeaderManifest(unittest.TestCase):
    def _checkpoint_with_headers(self, headers):
        checkpoint = object.__new__(Qwen35Checkpoint)
        checkpoint._headers = headers
        checkpoint._weight_map = {}
        return checkpoint

    def test_generated_manifest_has_exact_required_counts(self):
        self.assertEqual(len(EXPECTED_HEADERS), 119015)
        self.assertEqual(
            sum(
                spec.dtype == "F8_E4M3" and name.endswith(".weight")
                for name, spec in EXPECTED_HEADERS.items()
            ),
            130,
        )
        self.assertEqual(
            sum(name.endswith("weight_packed") for name in EXPECTED_HEADERS), 29184
        )
        self.assertEqual(
            sum(
                spec.dtype == "F16"
                and ".mlp.experts." in name
                and name.endswith(".weight")
                for name, spec in EXPECTED_HEADERS.items()
            ),
            1536,
        )

    def test_rejects_wrong_name_shape_and_dtype_before_payload_read(self):
        name = "model.language_model.layers.3.self_attn.q_proj.weight"
        spec = EXPECTED_HEADERS[name]
        for malformed in (
            {
                "model.language_model.layers.3.self_attn.typo.weight": (
                    spec.shape,
                    "F8_E4M3",
                )
            },
            {name: ((4096, 2048), "F8_E4M3")},
            {name: (spec.shape, "F16")},
            {
                "model.language_model.layers.3.self_attn.q_proj.weight_scale": (
                    (64, 15),
                    "F16",
                )
            },
        ):
            checkpoint = self._checkpoint_with_headers(malformed)
            with self.assertRaises(ValueError):
                checkpoint._validate_headers(malformed)
