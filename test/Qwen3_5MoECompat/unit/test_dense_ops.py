import unittest

import torch

from sglang.srt.layers.qwen3_5.dense import fp16_embedding, linear_fp16
from sglang.srt.layers.qwen3_5.ops import gated_rms_norm_silu, gemma_rms_norm, residual_add, sigmoid_mul
from sglang.srt.layers.qwen3_5.weights import Weight
from test.Qwen3_5MoECompat.reference.dense import gated_rms_norm_silu as ref_gated
from test.Qwen3_5MoECompat.reference.dense import gemma_rms_norm as ref_gemma
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase


class TestDenseOps(V100TestCase):
    def test_nt_gemm_sm70_shape_sweep(self):
        """The safe 32-cube Volta configuration must survive tails and CTAs."""
        torch.manual_seed(13)
        for m, n, k in ((1, 512, 2048), (17, 512, 2048), (128, 512, 2048), (33, 37, 257)):
            x = torch.randn((m, k), dtype=torch.float16, device="cuda")
            w = torch.randn((n, k), dtype=torch.float16, device="cuda")
            actual = linear_fp16(x, Weight("fp16", w, (n, k))).float()
            expected = x.float() @ w.float().t()
            nrmse = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
            self.assertLessEqual(nrmse.item(), 2e-3, (m, n, k, nrmse.item()))
        empty = torch.empty((0, 128), dtype=torch.float16, device="cuda")
        weight = torch.empty((64, 128), dtype=torch.float16, device="cuda")
        self.assertEqual(tuple(linear_fp16(empty, Weight("fp16", weight, (64, 128))).shape), (0, 64))

    def test_nt_gemm_and_embedding(self):
        torch.manual_seed(1)
        x = torch.randn((17, 2048), dtype=torch.float16, device="cuda")
        w = torch.randn((64, 2048), dtype=torch.float16, device="cuda")
        actual = linear_fp16(x, Weight("fp16", w, (64, 2048)))
        expected = (x.float() @ w.float().t()).half()
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-2)
        table = torch.randn((31, 2048), dtype=torch.float16, device="cuda")
        ids = torch.tensor([0, 30, 7], dtype=torch.int64, device="cuda")
        torch.testing.assert_close(fp16_embedding(ids, table), table[ids])
        with self.assertRaises(IndexError):
            fp16_embedding(torch.tensor([-1], dtype=torch.int64, device="cuda"), table, validate_ids=True)
        with self.assertRaises(IndexError):
            fp16_embedding(torch.tensor([31], dtype=torch.int64, device="cuda"), table, validate_ids=True)

    def test_norm_and_elementwise(self):
        torch.manual_seed(2)
        x = torch.randn((7, 2048), dtype=torch.float16, device="cuda")
        y = torch.randn_like(x)
        weight = torch.randn((2048,), dtype=torch.float16, device="cuda")
        torch.testing.assert_close(gemma_rms_norm(x, weight), ref_gemma(x, weight), rtol=3e-3, atol=5e-3)
        torch.testing.assert_close(gated_rms_norm_silu(x, y, weight), ref_gated(x, y, weight), rtol=3e-3, atol=5e-3)
        torch.testing.assert_close(residual_add(x, y), (x + y).half(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(sigmoid_mul(x, y), (x * y.sigmoid()).half(), rtol=3e-3, atol=2e-3)
        with self.assertRaises(ValueError):
            sigmoid_mul(x[:, ::2], y[:, ::2])
        with self.assertRaises(ValueError):
            linear_fp16(x[:, ::2], Weight("fp16", torch.randn((64, 1024), dtype=torch.float16, device="cuda"), (64, 1024)))
