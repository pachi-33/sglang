import os
from pathlib import Path
import unittest

from sglang.srt.layers.qwen3_5.checkpoint import Qwen35Checkpoint


MODEL_DIR = Path(os.environ.get("QWEN35_MODEL_DIR", "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16"))


class TestCheckpointManifest(unittest.TestCase):
    @unittest.skipUnless(MODEL_DIR.is_dir(), "Qwen3.5 checkpoint unavailable")
    def test_quantized_matrix_manifest(self):
        audit = Qwen35Checkpoint(MODEL_DIR).audit()
        self.assertEqual((audit.layers, audit.fp8_matrices, audit.nvfp4_matrices), (40, 130, 29184))
        self.assertEqual(audit.fp16_expert_matrices, 1536)


class TestWeightContract(unittest.TestCase):
    def test_fp16_disallows_quantization_metadata(self):
        import torch
        from sglang.srt.layers.qwen3_5.weights import Weight
        with self.assertRaises(ValueError):
            Weight("fp16", torch.zeros((2, 2), dtype=torch.float16), (2, 2), torch.ones(1))
