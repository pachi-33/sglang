import os
from pathlib import Path
import unittest

import torch

from sglang.srt.layers.qwen3_5.model_ops import (
    full_qk_rope_gate,
    gated_attention_fp8,
    gated_gdn_fp8,
)
from sglang.srt.layers.qwen3_5.ops import residual_add_gemma_rms_norm
from sglang.srt.layers.qwen3_5.quantization import quantize_fp8
from test.Qwen3_5MoECompat.reference.codec import quantize_a8
from test.Qwen3_5MoECompat.reference.codec import decode_e4m3fn
from test.Qwen3_5MoECompat.reference.dense import gemma_rms_norm
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase


class TestFullAttentionProducer(V100TestCase):
    def test_q_gate_head_layout_norm_and_rope(self):
        torch.manual_seed(19)
        tokens = 3
        qgate = torch.randn((tokens, 8192), device="cuda", dtype=torch.float16).contiguous()
        k = torch.randn((tokens, 512), device="cuda", dtype=torch.float16).contiguous()
        positions = torch.tensor([0, 7, 101], device="cuda", dtype=torch.int32)
        qw = torch.randn((256,), device="cuda", dtype=torch.float16)
        kw = torch.randn((256,), device="cuda", dtype=torch.float16)
        q, kout, gate = full_qk_rope_gate(qgate, k, positions, qw, kw)

        # Independent layout/reference: Q and gate alternate within each 512
        # columns, rather than occupying two 4096-wide projection halves.
        qref = qgate.reshape(tokens, 16, 512)[..., :256].float()
        gref = qgate.reshape(tokens, 16, 512)[..., 256:]
        kref = k.reshape(tokens, 2, 256).float()
        qref = (qref * torch.rsqrt(qref.square().mean(-1, keepdim=True) + 1e-6) * (1 + qw.float())).half().float()
        kref = (kref * torch.rsqrt(kref.square().mean(-1, keepdim=True) + 1e-6) * (1 + kw.float())).half().float()
        pair = torch.arange(32, device="cuda", dtype=torch.float32)
        angle = positions.float()[:, None, None] / torch.exp(pair[None, None] * torch.log(torch.tensor(10_000_000.0, device="cuda")) / 32)
        def rope(x):
            first, second = x[..., :32], x[..., 32:64]
            y = x.clone()
            y[..., :32] = first * torch.cos(angle) - second * torch.sin(angle)
            y[..., 32:64] = first * torch.sin(angle) + second * torch.cos(angle)
            return y.half()
        torch.cuda.synchronize()
        self.assertLess((q - rope(qref)).abs().max().item(), 2e-3)
        self.assertLess((kout - rope(kref)).abs().max().item(), 2e-3)
        self.assertTrue(torch.equal(gate, gref))

    def test_gated_attention_a8_matches_independent_fp16_boundary_codec(self):
        torch.manual_seed(29)
        attention = torch.randn((2, 16, 256), device="cuda", dtype=torch.float16).contiguous()
        gate = torch.randn_like(attention).contiguous()
        observed = gated_attention_fp8(attention, gate)
        # CPU codec deliberately does not share the Triton producer.  The
        # cast is the model's explicit sigmoid/multiply FP16 boundary.
        boundary = (attention.cpu().float() * torch.sigmoid(gate.cpu().float())).half()
        expected_data, expected_scale = quantize_a8(boundary.reshape(2, 4096))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(observed.data.cpu(), expected_data.cpu()))
        self.assertTrue(torch.equal(observed.block_scale.cpu(), expected_scale.cpu()))

    def test_gated_attention_zero_gate_is_exact_base_a8(self):
        """The fused local-scale arithmetic matches A8 at a boundary-heavy T17."""
        torch.manual_seed(179)
        attention = torch.randn((17, 16, 256), device="cuda", dtype=torch.float16).contiguous()
        gate = torch.zeros_like(attention)
        observed = gated_attention_fp8(attention, gate)
        base_data, base_scale = quantize_a8((attention.float() * 0.5).half().cpu().reshape(17, 4096))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(observed.data.cpu(), base_data))
        self.assertTrue(torch.equal(observed.block_scale.cpu(), base_scale))

    def test_gdn_norm_silu_a8_matches_independent_fp16_boundary_codec(self):
        torch.manual_seed(31)
        value = torch.randn((2, 32, 128), device="cuda", dtype=torch.float32).contiguous()
        z = torch.randn((2, 4096), device="cuda", dtype=torch.float16).contiguous()
        weight = torch.randn((128,), device="cuda", dtype=torch.float16).contiguous()
        observed = gated_gdn_fp8(value, z, weight)
        value_cpu = value.cpu().half().float()  # FP32 recurrent -> FP16 boundary
        norm = value_cpu * torch.rsqrt(value_cpu.square().mean(-1, keepdim=True) + 1e-6)
        boundary = (norm * weight.cpu().float()[None, None, :] * torch.nn.functional.silu(z.cpu().float()).reshape(2, 32, 128)).half()
        expected_data, expected_scale = quantize_a8(boundary.reshape(2, 4096))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(observed.data.cpu(), expected_data.cpu()))
        self.assertTrue(torch.equal(observed.block_scale.cpu(), expected_scale.cpu()))

    def test_fused_residual_norm_keeps_fp16_sum_boundary(self):
        torch.manual_seed(37)
        x = (torch.randn((3, 2048), device="cuda", dtype=torch.float16) * 0.3).contiguous()
        residual = (torch.randn_like(x) * 0.3).contiguous()
        weight = torch.randn((2048,), device="cuda", dtype=torch.float16).contiguous()
        summed, normalized = residual_add_gemma_rms_norm(x, residual, weight)
        expected_sum = (x.float() + residual.float()).half()
        expected_norm = (expected_sum.float() * torch.rsqrt(expected_sum.float().square().mean(-1, keepdim=True) + 1e-6)
                         * (1 + weight.float())).half()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(summed, expected_sum))
        self.assertTrue(torch.equal(normalized, expected_norm))

    @unittest.skipUnless(Path(os.environ.get("QWEN35_MODEL_DIR", "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16")).is_dir(), "real checkpoint unavailable")
    def test_real_layer3_full_attention_composition(self):
        """Independent CPU oracle for all Q/K/V/O W8 projections and residual."""
        from sglang.srt.layers.qwen3_5.runner import Qwen35StatelessRunner
        model_dir = os.environ.get("QWEN35_MODEL_DIR", "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16")
        runner = Qwen35StatelessRunner(model_dir, [3], load_globals=False)
        torch.manual_seed(43)
        hidden = (torch.randn((3, 2048), device="cuda", dtype=torch.float16) * 0.1).contiguous()
        pos = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
        observed = runner._full_attention(hidden, runner.layers[0], pos, cu, 2)
        w = runner.layers[0].weights
        def dequant(weight):
            raw = decode_e4m3fn(weight.data.cpu()).float()
            sf = weight.block_scale.cpu().float().repeat_interleave(128, 0).repeat_interleave(128, 1)
            return raw * sf[:raw.shape[0], :raw.shape[1]]
        def linear(x, weight):
            return (x.float() @ dequant(weight).t()).half()
        h = hidden.cpu()
        norm = gemma_rms_norm(h, w["input_layernorm.weight"].cpu())
        # The CPU codec is the oracle for both A8 producer boundaries.
        a8_data, a8_scale = quantize_a8(norm.reshape(3, 2048))
        adeq = decode_e4m3fn(a8_data).float() * a8_scale.repeat_interleave(128, 1)
        qg, kk, vv = (linear(adeq, w[name]) for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))
        qg = qg.reshape(3, 16, 512)
        q = qg[..., :256].float(); gate = qg[..., 256:].float(); k = kk.reshape(3, 2, 256).float()
        q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6) * (1 + w["self_attn.q_norm.weight"].cpu().float())).half().float()
        k = (k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6) * (1 + w["self_attn.k_norm.weight"].cpu().float())).half().float()
        d = torch.arange(32, dtype=torch.float32); angle = pos.cpu().float()[:, None, None] / torch.exp(d[None, None] * torch.log(torch.tensor(10_000_000.0)) / 32)
        def rope(x):
            y=x.clone(); a,b=x[...,:32],x[...,32:64]; y[...,:32]=a*torch.cos(angle)-b*torch.sin(angle); y[...,32:64]=a*torch.sin(angle)+b*torch.cos(angle); return y.half()
        q, k, vv = rope(q), rope(k), vv.reshape(3,2,256)
        attn = torch.empty((3,16,256), dtype=torch.float16)
        for start, end in ((0,2),(2,3)):
            for token in range(start,end):
                for head in range(16):
                    score=(q[token,head].float() @ k[start:token+1,head//8].float().t()) * (256**-0.5)
                    attn[token,head]=(torch.softmax(score,0).float() @ vv[start:token+1,head//8].float()).half()
        oa_data, oa_scale = quantize_a8((attn.float() * torch.sigmoid(gate)).half().reshape(3,4096))
        oinput = decode_e4m3fn(oa_data).float() * oa_scale.repeat_interleave(128,1)
        expected = linear(oinput, w["self_attn.o_proj"])
        torch.cuda.synchronize()
        error = (observed.cpu().float() - expected.float()).abs()
        self.assertLess((error.square().mean().sqrt() / expected.float().square().mean().sqrt()).item(), 5e-3)


if __name__ == "__main__":
    unittest.main()
