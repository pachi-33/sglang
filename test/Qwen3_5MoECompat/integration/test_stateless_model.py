import os
from pathlib import Path
import unittest

import torch

from sglang.srt.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase


MODEL_DIR = Path(os.environ.get("QWEN35_MODEL_DIR", "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16"))


class TestStatelessFourLayerSlice(V100TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not MODEL_DIR.is_dir():
            raise unittest.SkipTest("real checkpoint unavailable")
        cls.model = Qwen3_5MoeForConditionalGeneration.from_checkpoint(MODEL_DIR, selected_layer_ids=range(4))

    def test_t1_default_last_token_logits(self):
        hidden, logits = self.model.forward_no_cache(
            input_ids=torch.tensor([1], device="cuda", dtype=torch.int32),
            positions=torch.tensor([0], device="cuda", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int32), max_seqlen=1,
        )
        torch.cuda.synchronize()
        self.assertEqual((tuple(hidden.shape), tuple(logits.shape)), ((1, 2048), (1, 248320)))
        self.assertTrue(torch.isfinite(hidden).all() and torch.isfinite(logits).all())

    def test_ragged_explicit_logits_indices(self):
        hidden, logits = self.model.forward_no_cache(
            input_ids=torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32),
            positions=torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32), max_seqlen=2,
            logits_indices=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        )
        torch.cuda.synchronize()
        self.assertEqual((tuple(hidden.shape), tuple(logits.shape)), ((3, 2048), (2, 248320)))
        self.assertTrue(torch.isfinite(hidden).all() and torch.isfinite(logits).all())
