import math
import unittest

import torch

from sglang.srt.layers.qwen3_5.attention import causal_gqa, normalize_and_rope_qk, partial_neox_rope
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase


def reference_packed_gqa(q, k, v, cu):
    out = torch.empty_like(q)
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        for head in range(q.shape[1]):
            kv_head = head // (q.shape[1] // k.shape[1])
            scores = q[start:end, head].float() @ k[start:end, kv_head].float().t() / math.sqrt(q.shape[-1])
            scores.masked_fill_(torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1), float("-inf"))
            out[start:end, head] = (scores.softmax(-1) @ v[start:end, kv_head].float()).half()
    return out


class TestCausalGQA(V100TestCase):
    def assert_attention_matches_reference(self, lengths, seed):
        torch.manual_seed(seed)
        tokens = sum(lengths)
        q = torch.randn((tokens, 16, 256), dtype=torch.float16, device="cuda")
        k = torch.randn((tokens, 2, 256), dtype=torch.float16, device="cuda")
        v = torch.randn((tokens, 2, 256), dtype=torch.float16, device="cuda")
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
        actual = causal_gqa(q, k, v, cu, max_seqlen=max(lengths)).float()
        expected = reference_packed_gqa(q, k, v, cu.cpu()).cuda().float()
        nrmse = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        self.assertLessEqual(nrmse.item(), 5e-3)
        for start in cu[:-1].cpu().tolist():
            for head in range(16):
                torch.testing.assert_close(actual[start, head], v[start, head // 8].float(), rtol=0, atol=0)

    def test_tensorcore_attention_single_sequences_through_2048(self):
        for seed, length in enumerate((1, 17, 65, 129, 2048), start=19):
            with self.subTest(length=length):
                self.assert_attention_matches_reference([length], seed)

    def test_tensorcore_attention_ragged_total_2048(self):
        self.assert_attention_matches_reference([1, 17, 65, 129, 1836], 31)

    def test_tensorcore_attention_multitile(self):
        torch.manual_seed(19)
        q = torch.randn((33, 16, 256), dtype=torch.float16, device="cuda")
        k = torch.randn((33, 2, 256), dtype=torch.float16, device="cuda")
        v = torch.randn((33, 2, 256), dtype=torch.float16, device="cuda")
        cu = torch.tensor([0, 17, 33], dtype=torch.int32, device="cuda")
        actual = causal_gqa(q, k, v, cu, max_seqlen=17).float()
        expected = reference_packed_gqa(q, k, v, cu.cpu()).cuda().float()
        nrmse = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        self.assertLessEqual(nrmse.item(), 5e-3)

    def test_partial_neox_rope(self):
        torch.manual_seed(18)
        x = torch.randn((3, 2, 256), dtype=torch.float16, device="cuda")
        positions = torch.tensor([0, 1, 9], dtype=torch.int64, device="cuda")
        actual = partial_neox_rope(x, positions)
        expected = x.clone()
        pair = torch.arange(32, device="cuda", dtype=torch.float32)
        angles = positions.float()[:, None] / (10_000_000.0 ** (pair / 32))
        c, s = angles.cos()[:, None], angles.sin()[:, None]
        expected[:, :, :32] = (x[:, :, :32].float() * c - x[:, :, 32:64].float() * s).half()
        expected[:, :, 32:64] = (x[:, :, :32].float() * s + x[:, :, 32:64].float() * c).half()
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-3)

    def test_ragged_causal_gqa(self):
        self.assert_attention_matches_reference([3, 4], 17)

    def test_nonempty_int64_cu_seqlens(self):
        torch.manual_seed(52)
        q = torch.randn((5, 16, 256), dtype=torch.float16, device="cuda")
        k = torch.randn((5, 2, 256), dtype=torch.float16, device="cuda")
        v = torch.randn_like(k)
        cu = torch.tensor([0, 2, 5], dtype=torch.int64, device="cuda")
        actual = causal_gqa(q, k, v, cu, max_seqlen=3)
        expected = reference_packed_gqa(q, k, v, cu.cpu()).cuda()
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-2)

    def test_empty_attention_honors_output_identity(self):
        q = torch.empty((0, 16, 256), dtype=torch.float16, device="cuda")
        k = torch.empty((0, 2, 256), dtype=torch.float16, device="cuda")
        out = torch.empty_like(q)
        cu = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        self.assertIs(causal_gqa(q, k, k, cu, max_seqlen=0, out=out), out)

    def test_attention_rejects_invalid_metadata_without_launch(self):
        q = torch.empty((0, 16, 256), dtype=torch.float16, device="cuda")
        k = torch.empty((0, 2, 256), dtype=torch.float16, device="cuda")
        cu = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(ValueError, "rank-1"):
            causal_gqa(q, k, k, cu[None], max_seqlen=0)
        with self.assertRaisesRegex(ValueError, "int32 or int64"):
            causal_gqa(q, k, k, cu.to(torch.float16), max_seqlen=0)
        with self.assertRaisesRegex(TypeError, "Python int"):
            causal_gqa(q, k, k, cu, max_seqlen=torch.tensor(0))

    def test_rope_empty_and_invalid_metadata(self):
        x = torch.empty((0, 2, 256), dtype=torch.float16, device="cuda")
        positions = torch.empty((0,), dtype=torch.int64, device="cuda")
        out = torch.empty_like(x)
        self.assertIs(partial_neox_rope(x, positions, out=out), out)
        with self.assertRaisesRegex(TypeError, "FP16"):
            partial_neox_rope(x.float(), positions)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            partial_neox_rope(x, positions, theta=0.0)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            nonempty = torch.empty((1, 2, 256), dtype=torch.float16, device="cuda")
            partial_neox_rope(nonempty, torch.zeros((1,), dtype=torch.int64, device="cuda"), out=nonempty)

    def test_normalize_and_rope_qk_and_layout_contract(self):
        torch.manual_seed(44)
        q = torch.randn((2, 2, 256), dtype=torch.float16, device="cuda")
        k = torch.randn((2, 2, 256), dtype=torch.float16, device="cuda")
        qw = torch.randn((256,), dtype=torch.float16, device="cuda")
        kw = torch.randn((256,), dtype=torch.float16, device="cuda")
        positions = torch.zeros((2,), dtype=torch.int64, device="cuda")
        actual_q, actual_k = normalize_and_rope_qk(q, k, qw, kw, positions)
        expected_q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6) * (1 + qw.float())).half()
        expected_k = (k.float() * torch.rsqrt(k.float().square().mean(-1, keepdim=True) + 1e-6) * (1 + kw.float())).half()
        torch.testing.assert_close(actual_q, expected_q, rtol=3e-3, atol=5e-3)
        torch.testing.assert_close(actual_k, expected_k, rtol=3e-3, atol=5e-3)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            normalize_and_rope_qk(q.transpose(0, 1), k, qw, kw, positions)
