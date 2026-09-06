import types
import unittest
from pathlib import Path
from test.Qwen3_5MoECompat.reference.codec import quantize_a4, unpack_a4
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase

import torch

from sglang.srt.layers.qwen3_5.checkpoint import Qwen35Checkpoint
from sglang.srt.layers.qwen3_5.moe import (
    MoeWeights,
    _build_dispatch,
    _fp16_sigmoid_multiply,
    _fp16_swiglu,
    _grouped_gemm,
    _paired_gemm1_swiglu_a4,
    execute_experts,
    execute_experts_unfused_baseline,
    execute_fp16_experts,
    fused_moe,
    route_topk,
)
from sglang.srt.layers.qwen3_5.quantization import quantize_nvfp4
from sglang.srt.layers.qwen3_5.weights import Weight

REAL_NVFP4_CHECKPOINT = Path(
    "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16"
)


class TestQwen35Moe(V100TestCase):
    def test_topk_is_fp32_and_sorted(self):
        logits = torch.full((2, 256), -10.0, dtype=torch.float16, device="cuda")
        logits[0, :4] = torch.tensor(
            [0.0, 3.0, 2.0, 1.0], dtype=torch.float16, device="cuda"
        )
        logits[1] = 0.0
        ids, weights = route_topk(logits, top_k=8)
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(ids[:, :3].tolist(), [[1, 2, 3], [0, 1, 2]])
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, device="cuda"))

    def test_fp16_routed_moe_fallback(self):
        x = torch.tensor([[1.0, -1.0]], dtype=torch.float16, device="cuda")
        gate_data = torch.zeros((256, 4, 2), dtype=torch.float16, device="cuda")
        down_data = torch.zeros((256, 2, 2), dtype=torch.float16, device="cuda")
        gate_data[0] = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
            dtype=torch.float16,
            device="cuda",
        )
        down_data[0] = torch.eye(2, dtype=torch.float16, device="cuda")
        gate_up = types.SimpleNamespace(data=gate_data)
        down = types.SimpleNamespace(data=down_data)
        out = execute_fp16_experts(
            x,
            gate_up,
            down,
            torch.zeros((1, 8), dtype=torch.int32, device="cuda"),
            torch.full((1, 8), 0.125, dtype=torch.float32, device="cuda"),
        )
        expected = (
            torch.nn.functional.silu(
                torch.tensor([[1.0, -1.0]], dtype=torch.float16, device="cuda").float()
            )
            * torch.tensor([[2.0, -2.0]], dtype=torch.float16, device="cuda").float()
        ).half()
        self.assertTrue(torch.allclose(out, expected, atol=2e-3, rtol=2e-3))

    def test_routed_validators_reject_malformed_payloads_before_launch(self):
        ids = torch.zeros((1, 8), dtype=torch.int32, device="cuda")
        route_weights = torch.full((1, 8), 0.125, dtype=torch.float32, device="cuda")
        fp_gate = types.SimpleNamespace(
            data=torch.zeros((256, 4, 2), dtype=torch.float16, device="cuda")
        )
        fp_down = types.SimpleNamespace(
            data=torch.zeros((256, 2, 2), dtype=torch.float16, device="cuda")
        )
        x2 = torch.zeros((1, 2), dtype=torch.float16, device="cuda")
        with self.assertRaises(ValueError):
            execute_fp16_experts(
                torch.zeros((1, 4), dtype=torch.float16, device="cuda")[:, ::2],
                fp_gate,
                fp_down,
                ids,
                route_weights,
            )
        with self.assertRaises(ValueError):
            execute_fp16_experts(
                x2,
                fp_gate,
                types.SimpleNamespace(
                    data=torch.zeros((256, 1, 2), dtype=torch.float16, device="cuda")
                ),
                ids,
                route_weights,
            )
        with self.assertRaises(ValueError):
            execute_fp16_experts(x2, fp_gate, fp_down, ids, route_weights.half())
        noncontig_gate = types.SimpleNamespace(
            data=torch.zeros((256, 4, 4), dtype=torch.float16, device="cuda")[:, :, ::2]
        )
        with self.assertRaises(ValueError):
            execute_fp16_experts(x2, noncontig_gate, fp_down, ids, route_weights)
        empty_ids = torch.empty((0, 8), dtype=torch.int32, device="cuda")
        empty_weights = torch.empty((0, 8), dtype=torch.float32, device="cuda")
        fp_empty, fp_routes, fp_inverse = execute_fp16_experts(
            torch.empty((0, 2), dtype=torch.float16, device="cuda"),
            fp_gate,
            fp_down,
            empty_ids,
            empty_weights,
            capture_routes=True,
        )
        self.assertEqual(
            (tuple(fp_empty.shape), tuple(fp_routes.shape), tuple(fp_inverse.shape)),
            ((0, 2), (0, 2), (0,)),
        )
        self.assertEqual(fp_routes.dtype, torch.float32)
        with self.assertRaises(ValueError):
            fused_moe(
                x2,
                MoeWeights(
                    router=types.SimpleNamespace(
                        data=torch.zeros((256, 2), dtype=torch.float16, device="cuda")
                    ),
                    gate_up=fp_gate,
                    down=fp_down,
                ),
                residual=x2.float(),
            )
        tiny_moe = MoeWeights(
            router=types.SimpleNamespace(
                data=torch.zeros((256, 2), dtype=torch.float16, device="cuda")
            ),
            gate_up=fp_gate,
            down=fp_down,
        )
        with self.assertRaises(ValueError):
            fused_moe(
                x2,
                tiny_moe,
                residual=torch.zeros((1, 4), dtype=torch.float16, device="cuda")[
                    :, ::2
                ],
            )
        with self.assertRaises(ValueError):
            fused_moe(x2, tiny_moe, residual=torch.zeros((1, 2), dtype=torch.float16))

        nv_gate_cpu = torch.zeros((256, 64, 32), dtype=torch.float16)
        nv_down_cpu = torch.zeros((256, 32, 32), dtype=torch.float16)
        nv_gate, nv_down = _nvfp4_weight(nv_gate_cpu, device="cuda"), _nvfp4_weight(
            nv_down_cpu, device="cuda"
        )
        with self.assertRaises(ValueError):
            execute_experts(
                torch.zeros((1, 30), dtype=torch.float16, device="cuda"),
                nv_gate,
                nv_down,
                ids,
                route_weights,
            )
        with self.assertRaises(ValueError):
            execute_experts(
                torch.zeros((1, 64), dtype=torch.float16, device="cuda")[:, ::2],
                nv_gate,
                nv_down,
                ids,
                route_weights,
            )
        with self.assertRaises(ValueError):
            execute_experts(
                torch.zeros((1, 32), dtype=torch.float16, device="cuda"),
                nv_gate,
                nv_down,
                ids,
                torch.full((1, 16), 0.125, dtype=torch.float32, device="cuda")[:, ::2],
            )
        nv_empty, nv_routes, nv_inverse = execute_experts(
            torch.empty((0, 32), dtype=torch.float16, device="cuda"),
            nv_gate,
            nv_down,
            empty_ids,
            empty_weights,
            capture_routes=True,
        )
        self.assertEqual(
            (tuple(nv_empty.shape), tuple(nv_routes.shape), tuple(nv_inverse.shape)),
            ((0, 32), (0, 32), (0,)),
        )
        self.assertEqual(nv_routes.dtype, torch.float32)


def _nvfp4_weight(values: torch.Tensor, *, device: str) -> Weight:
    """Build a small raw expert batch through the independent test codec."""
    packed, local_scale = [], []
    for expert in values:
        data, scale, _ = quantize_a4(expert, torch.tensor([1.0]))
        packed.append(data)
        local_scale.append(scale)
    experts, n, k = values.shape
    return Weight(
        "nvfp4",
        torch.stack(packed).to(device),
        (experts, n, k),
        torch.stack(local_scale).to(device),
        torch.ones((experts,), dtype=torch.float32, device=device),
        torch.ones((experts,), dtype=torch.float32, device=device),
    )


class TestQwen35MoeGPU(V100TestCase):
    def test_gpu_dispatch_matches_stable_reference(self):
        for tokens, hotspot in (
            (0, False),
            (1, False),
            (17, False),
            (32, True),
            (128, False),
            (2048, True),
        ):
            with self.subTest(tokens=tokens, hotspot=hotspot):
                routes = tokens * 8
                flat = (
                    torch.zeros((routes,), dtype=torch.int32)
                    if hotspot
                    else torch.arange(routes, dtype=torch.int32) % 256
                )
                ids = flat.reshape(tokens, 8).cuda()
                source, positions, blocks = _build_dispatch(ids, 256)
                max_blocks = (routes + 31) // 32 + 256
                expected_source = torch.full((max_blocks * 32,), -1, dtype=torch.int32)
                expected_positions = torch.full(
                    (max_blocks * 32,), -1, dtype=torch.int32
                )
                expected_blocks = torch.full((max_blocks,), -1, dtype=torch.int32)
                cursor = 0
                for expert in range(256):
                    selected = torch.nonzero(flat == expert, as_tuple=False).flatten()
                    padded = ((selected.numel() + 31) // 32) * 32
                    if padded:
                        expected_source[
                            cursor * 32 : cursor * 32 + selected.numel()
                        ] = (selected // 8)
                        expected_positions[
                            cursor * 32 : cursor * 32 + selected.numel()
                        ] = selected
                        expected_blocks[cursor : cursor + padded // 32] = expert
                        cursor += padded // 32
                self.assertTrue(torch.equal(source.cpu(), expected_source))
                self.assertTrue(torch.equal(positions.cpu(), expected_positions))
                self.assertTrue(torch.equal(blocks.cpu(), expected_blocks))

    def test_routed_nvfp4_moe_is_deterministic(self):
        torch.manual_seed(19)
        # This uses two selected experts, exercises expert-major padding and
        # both grouped W4A4 GEMMs at the Qwen routed-expert matrix shapes
        # without allocating model-scale tensors.
        x_cpu = torch.randn((2, 32), dtype=torch.float16)
        gate_up_cpu = torch.randn((256, 64, 32), dtype=torch.float16) * 0.02
        down_cpu = torch.randn((256, 32, 32), dtype=torch.float16) * 0.02
        gate_up = _nvfp4_weight(gate_up_cpu, device="cuda")
        down = _nvfp4_weight(down_cpu, device="cuda")
        ids = torch.arange(16, dtype=torch.int32, device="cuda").reshape(2, 8)
        probs = torch.full((2, 8), 0.125, dtype=torch.float32, device="cuda")

        first = execute_experts(x_cpu.cuda(), gate_up, down, ids, probs)
        second = execute_experts(x_cpu.cuda(), gate_up, down, ids, probs)
        baseline = execute_experts_unfused_baseline(
            x_cpu.cuda(), gate_up, down, ids, probs
        )
        captured, route_values, inverse = execute_experts(
            x_cpu.cuda(), gate_up, down, ids, probs, capture_routes=True
        )
        self.assertEqual(first.dtype, torch.float16)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, captured))
        self.assertTrue(
            torch.equal(
                captured, route_values[inverse].reshape(2, 8, 32).sum(dim=1).half()
            )
        )
        self.assertTrue(torch.isfinite(first).all())
        nrmse = (
            (first.float() - baseline.float()).square().mean().sqrt()
            / baseline.float().square().mean().sqrt()
        ).item()
        self.assertLess(nrmse, 5e-3)

    def test_paired_gemm1_swiglu_a4_matches_projection_and_codec(self):
        """Validate the fused boundary without comparing separately-rounded GEMMs' bytes."""
        torch.manual_seed(23)
        x = (torch.randn((2, 32), dtype=torch.float16) * 0.1).cuda()
        gate_up_cpu = torch.randn((256, 64, 32), dtype=torch.float16) * 0.03
        down_cpu = torch.randn((256, 32, 32), dtype=torch.float16) * 0.03
        gate_base = _nvfp4_weight(gate_up_cpu, device="cuda")
        down_base = _nvfp4_weight(down_cpu, device="cuda")
        gate_g = torch.full((256,), 0.75, dtype=torch.float32, device="cuda")
        weight_g = torch.full((256,), 1.25, dtype=torch.float32, device="cuda")
        # The down activation multiplier is physical per expert and must not
        # accidentally be treated as a scalar or inherited from GEMM1.
        down_g = torch.linspace(0.55, 1.45, 256, dtype=torch.float32, device="cuda")
        gate_up = Weight(
            "nvfp4",
            gate_base.data,
            gate_base.logical_shape,
            gate_base.block_scale,
            weight_g,
            gate_g,
        )
        down = Weight(
            "nvfp4",
            down_base.data,
            down_base.logical_shape,
            down_base.block_scale,
            down_base.global_scale,
            down_g,
        )
        qx = quantize_nvfp4(x, gate_g[0])
        # 82 tiles on an 80-CTA persistent grid gives two CTAs a second tile,
        # exercising scratch reuse rather than only the one-tile fast path.
        blocks = torch.tensor([3, 17] * 41, dtype=torch.int32, device="cuda")
        source = (
            torch.arange(blocks.numel(), dtype=torch.int32).repeat_interleave(32) & 1
        ).cuda()
        qz, captured_z = _paired_gemm1_swiglu_a4(
            qx, source, blocks, gate_up, down, capture_z=True
        )
        torch.cuda.synchronize()

        x_decoded = unpack_a4(
            qx.data.cpu(), qx.block_scale.cpu(), gate_g[:1].cpu()
        ).float()
        expected = []
        for expert, token in ((3, 0), (17, 1)):
            w = unpack_a4(
                gate_up.data[expert].cpu(),
                gate_up.block_scale[expert].cpu(),
                weight_g[expert : expert + 1].cpu(),
            ).float()
            gu = (x_decoded[token : token + 1] @ w.T).half()
            z = (
                torch.nn.functional.silu(gu[:, :32].float()) * gu[:, 32:].float()
            ).half()
            expected.append(z)
        expected_z = torch.cat(expected).cuda()
        # Each pair of expert blocks has a different token/expert oracle.
        observed_z = torch.cat((captured_z[:1], captured_z[32:33]))
        nrmse = (
            (observed_z.float() - expected_z.float()).square().mean().sqrt()
            / expected_z.float().square().mean().sqrt()
        ).item()
        self.assertLess(nrmse, 2e-3)

        expert_rows = blocks.repeat_interleave(32).long()
        ref_data, ref_sf, _ = quantize_a4(captured_z.cpu(), down_g[expert_rows].cpu())
        self.assertTrue(torch.equal(qz.data.cpu(), ref_data))
        self.assertTrue(torch.equal(qz.block_scale.cpu(), ref_sf))

    def test_raw_grouped_gemm_unweighted_multiexpert_baseline(self):
        """Exercise the unweighted store path without placeholder route loads."""
        torch.manual_seed(31)
        x = (torch.randn((2, 32), dtype=torch.float16) * 0.1).cuda()
        weight = _nvfp4_weight(
            torch.randn((256, 32, 32), dtype=torch.float16) * 0.03, device="cuda"
        )
        qx = quantize_nvfp4(x, weight.input_global_scale[0])
        source = torch.cat(
            (torch.zeros(32, dtype=torch.int32), torch.ones(32, dtype=torch.int32))
        ).cuda()
        blocks = torch.tensor([3, 17], dtype=torch.int32, device="cuda")
        out = _grouped_gemm(qx, source, blocks, weight, weight.input_global_scale)
        torch.cuda.synchronize()
        x_decoded = unpack_a4(
            qx.data.cpu(), qx.block_scale.cpu(), weight.input_global_scale[:1].cpu()
        ).float()
        expected = []
        for expert, token in ((3, 0), (17, 1)):
            decoded_weight = unpack_a4(
                weight.data[expert].cpu(),
                weight.block_scale[expert].cpu(),
                weight.global_scale[expert : expert + 1].cpu(),
            ).float()
            expected.append(
                (x_decoded[token : token + 1] @ decoded_weight.T).half().expand(32, -1)
            )
        expected = torch.cat(expected).cuda()
        nrmse = (
            (out.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt()
        ).item()
        self.assertLess(nrmse, 2e-3)

    @unittest.skipUnless(
        REAL_NVFP4_CHECKPOINT.is_dir(), "real Qwen3.5 NVFP4 checkpoint is unavailable"
    )
    def test_real_layer1_all_experts_balanced(self):
        """Stream an independent raw-codec oracle for production layer 1."""
        layer = Qwen35Checkpoint(REAL_NVFP4_CHECKPOINT).load_layer(1, device="cuda")
        gate_up, down = (
            layer["mlp.experts.gate_up_proj"],
            layer["mlp.experts.down_proj"],
        )
        torch.manual_seed(53)
        x = torch.randn((32, 2048), dtype=torch.float16, device="cuda") * 0.1
        ids = torch.arange(256, dtype=torch.int32, device="cuda").reshape(32, 8)
        probs = torch.full((32, 8), 0.125, dtype=torch.float32, device="cuda")
        out = execute_experts(x, gate_up, down, ids, probs)
        # Decode/quantize one expert at a time through the independent test
        # codec.  This is deliberately streamed: it never expands a full
        # [256,N,K] reference weight batch.
        x_data, x_sf, _ = quantize_a4(x.cpu(), gate_up.input_global_scale[:1].cpu())
        x_decoded = (
            unpack_a4(x_data, x_sf, gate_up.input_global_scale[:1].cpu()).float().cuda()
        )
        expected = torch.zeros((32, 2048), dtype=torch.float32, device="cuda")
        for expert in range(256):
            token = expert // 8
            gate_up_decoded = (
                unpack_a4(
                    gate_up.data[expert].cpu(),
                    gate_up.block_scale[expert].cpu(),
                    gate_up.global_scale[expert : expert + 1].cpu(),
                )
                .float()
                .cuda()
            )
            gu = (x_decoded[token : token + 1] @ gate_up_decoded.T).half()
            z = (
                torch.nn.functional.silu(gu[:, :512].float()) * gu[:, 512:].float()
            ).half()
            z_data, z_sf, _ = quantize_a4(
                z.cpu(), down.input_global_scale[expert : expert + 1].cpu()
            )
            z_decoded = (
                unpack_a4(
                    z_data, z_sf, down.input_global_scale[expert : expert + 1].cpu()
                )
                .float()
                .cuda()
            )
            down_decoded = (
                unpack_a4(
                    down.data[expert].cpu(),
                    down.block_scale[expert].cpu(),
                    down.global_scale[expert : expert + 1].cpu(),
                )
                .float()
                .cuda()
            )
            expected[token : token + 1] += (
                z_decoded @ down_decoded.T
            ).half().float() * 0.125
        torch.cuda.synchronize()
        expected = expected.half()
        error = (out.float() - expected.float()).abs()
        nrmse = (
            error.square().mean().sqrt() / expected.float().square().mean().sqrt()
        ).item()
        maximum = error.max().item()
        p99 = torch.quantile(error.flatten(), 0.99).item()
        self.assertTrue(torch.isfinite(out).all(), (nrmse, maximum, p99))
        self.assertLess(nrmse, 5e-3, (nrmse, maximum, p99))

    @unittest.skipUnless(
        REAL_NVFP4_CHECKPOINT.is_dir(), "real Qwen3.5 NVFP4 checkpoint is unavailable"
    )
    def test_real_layer0_fp16_all_experts_balanced_precision(self):
        """The FP16 routed fallback keeps the same all-expert Top-8 semantics."""
        layer = Qwen35Checkpoint(REAL_NVFP4_CHECKPOINT).load_layer(0, device="cuda")
        gate_up, down = (
            layer["mlp.experts.gate_up_proj"],
            layer["mlp.experts.down_proj"],
        )
        torch.manual_seed(41)
        x = torch.randn((32, 2048), dtype=torch.float16, device="cuda") * 0.1
        ids = torch.arange(256, dtype=torch.int32, device="cuda").reshape(32, 8)
        probs = torch.full((32, 8), 0.125, dtype=torch.float32, device="cuda")
        out = execute_fp16_experts(x, gate_up, down, ids, probs)
        # Independent test oracle: each expert occurs exactly once, so this is
        # the direct per-route FP16 definition before Top-8 accumulation.
        expected = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(256):
            token = expert // 8
            gu = (x[token : token + 1].float() @ gate_up.data[expert].float().T).half()
            z = (
                torch.nn.functional.silu(gu[:, :512].float()) * gu[:, 512:].float()
            ).half()
            expected[token : token + 1] += (
                z.float() @ down.data[expert].float().T
            ).half().float() * 0.125
        torch.cuda.synchronize()
        expected = expected.half()
        nrmse = (
            (out.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt()
        ).item()
        self.assertLess(nrmse, 5e-3)

    @unittest.skipUnless(
        REAL_NVFP4_CHECKPOINT.is_dir(), "real Qwen3.5 NVFP4 checkpoint is unavailable"
    )
    def test_real_shared_expert_gate_fp16_sigmoid_boundary(self):
        """Match CUDA's FP16 sigmoid boundary before shared-output multiply."""
        layer = Qwen35Checkpoint(REAL_NVFP4_CHECKPOINT).load_layer(1, device="cuda")
        torch.manual_seed(67)
        x = torch.randn((3, 2048), dtype=torch.float16, device="cuda") * 0.1
        shared = _fp16_swiglu(
            x,
            layer["mlp.shared_expert.gate_up_proj"],
            layer["mlp.shared_expert.down_proj"],
        )
        gate = x.float() @ layer["mlp.shared_expert_gate"].data.float().T
        observed = _fp16_sigmoid_multiply(shared, gate.half())
        expected_gate = torch.sigmoid(gate.half().float()).half()
        expected = (shared.float() * expected_gate.float()).half()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(observed, expected))

    @unittest.skipUnless(
        REAL_NVFP4_CHECKPOINT.is_dir(), "real Qwen3.5 NVFP4 checkpoint is unavailable"
    )
    def test_real_layer1_fused_moe_single_token_oracle(self):
        """End-to-end wrapper check: stable router, routed experts, shared gate, residual."""
        layer = Qwen35Checkpoint(REAL_NVFP4_CHECKPOINT).load_layer(1, device="cuda")
        torch.manual_seed(71)
        x = torch.randn((1, 2048), dtype=torch.float16, device="cuda") * 0.1
        moe = MoeWeights(
            router=layer["mlp.gate"],
            gate_up=layer["mlp.experts.gate_up_proj"],
            down=layer["mlp.experts.down_proj"],
            shared_gate_up=layer["mlp.shared_expert.gate_up_proj"],
            shared_down=layer["mlp.shared_expert.down_proj"],
            shared_gate=layer["mlp.shared_expert_gate"],
        )
        out = fused_moe(x, moe)
        logits = (x.float() @ layer["mlp.gate"].data.float().T).half().float()
        probs = torch.softmax(logits, dim=-1)
        expert_ids = torch.argsort(probs, dim=-1, descending=True, stable=True)[0, :8]
        route_weights = probs[0, expert_ids]
        route_weights = route_weights / route_weights.sum()
        gate_up, down = moe.gate_up, moe.down
        x_data, x_sf, _ = quantize_a4(x.cpu(), gate_up.input_global_scale[:1].cpu())
        x_decoded = (
            unpack_a4(x_data, x_sf, gate_up.input_global_scale[:1].cpu()).float().cuda()
        )
        routed = torch.zeros((1, 2048), dtype=torch.float32, device="cuda")
        for expert, route_weight in zip(expert_ids.tolist(), route_weights.tolist()):
            w1 = (
                unpack_a4(
                    gate_up.data[expert].cpu(),
                    gate_up.block_scale[expert].cpu(),
                    gate_up.global_scale[expert : expert + 1].cpu(),
                )
                .float()
                .cuda()
            )
            gu = (x_decoded @ w1.T).half()
            z = (
                torch.nn.functional.silu(gu[:, :512].float()) * gu[:, 512:].float()
            ).half()
            z_data, z_sf, _ = quantize_a4(
                z.cpu(), down.input_global_scale[expert : expert + 1].cpu()
            )
            z_decoded = (
                unpack_a4(
                    z_data, z_sf, down.input_global_scale[expert : expert + 1].cpu()
                )
                .float()
                .cuda()
            )
            w2 = (
                unpack_a4(
                    down.data[expert].cpu(),
                    down.block_scale[expert].cpu(),
                    down.global_scale[expert : expert + 1].cpu(),
                )
                .float()
                .cuda()
            )
            routed += (z_decoded @ w2.T).half().float() * route_weight
        routed = routed.half()
        shared_gu = (x.float() @ moe.shared_gate_up.data.float().T).half()
        shared_z = (
            torch.nn.functional.silu(shared_gu[:, :512].float())
            * shared_gu[:, 512:].float()
        ).half()
        shared = (shared_z.float() @ moe.shared_down.data.float().T).half()
        shared_gate = (x.float() @ moe.shared_gate.data.float().T).half()
        shared = (
            shared.float() * torch.sigmoid(shared_gate.float()).half().float()
        ).half()
        expected = (routed.float() + shared.float()).half()
        torch.cuda.synchronize()
        nrmse = (
            (out.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt()
        ).item()
        self.assertLess(nrmse, 5e-3)


if __name__ == "__main__":
    import unittest

    unittest.main()
