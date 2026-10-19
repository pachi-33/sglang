"""CUDA-graph replay coverage for the MoE trace recorder."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.moe_trace.codec import quantize_and_pack_int4
from sglang.srt.moe_trace.config import MoeTraceConfig
from sglang.srt.moe_trace.recorder import MoeTraceRecorder
from sglang.srt.moe_trace.types import MoeTraceSite
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class _DecodeMode:
    @staticmethod
    def is_decode():
        return True


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMoeTraceRecorderCuda(unittest.TestCase):
    def test_cuda_graph_replay_updates_flat_trace_buffers(self):
        with tempfile.TemporaryDirectory() as root:
            config = MoeTraceConfig(
                output_dir=Path(root),
                expert_routes=True,
                router_inputs=True,
                max_decode_tokens=0,
                activation_group_size=64,
                queue_depth=2,
                overflow_policy="block",
            )
            site = MoeTraceSite(0, "layers.0.topk", 0, "topk", 131, 8, 2)
            with patch(
                "sglang.srt.moe_trace.recorder.get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ):
                recorder = MoeTraceRecorder(config, [site], max_rows=2, device="cuda")
            module = nn.Module()
            module._moe_trace_site_id = 0
            hidden = torch.randn((2, 131), dtype=torch.float16, device="cuda")
            ids = torch.tensor([[1, 3], [2, 7]], dtype=torch.int32, device="cuda")
            weights = torch.tensor(
                [[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32, device="cuda"
            )

            # Compile/autotune all kernels away from graph capture.
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream), recorder.capture_scope():
                recorder.capture_router_input(module, hidden)
                recorder.capture_route(module, ids, weights)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            recorder.begin_forward(True)
            with recorder.capture_scope(), torch.cuda.graph(graph):
                recorder.capture_router_input(module, hidden)
                recorder.capture_route(module, ids, weights)

            # Exact integer multiples of each group's scale avoid making this
            # graph-lifecycle test depend on CPU-vs-GPU behavior at a .5 tie.
            q_values = (torch.arange(131, dtype=torch.int32) % 15 - 7).to(torch.float32)
            q_values[128:] = torch.tensor([-7.0, 0.0, 7.0])
            updated_hidden = torch.stack((q_values * 0.5, q_values * 0.25)).to(
                device="cuda", dtype=torch.float16
            )
            updated_ids = torch.tensor(
                [[7, 0], [4, 5]], dtype=torch.int32, device="cuda"
            )
            updated_weights = torch.tensor(
                [[0.9, 0.1], [0.55, 0.45]], dtype=torch.float32, device="cuda"
            )
            hidden.copy_(updated_hidden)
            ids.copy_(updated_ids)
            weights.copy_(updated_weights)
            recorder.begin_forward(True)
            graph.replay()

            batch = SimpleNamespace(
                forward_mode=_DecodeMode(),
                rids=["r0", "r1"],
                batch_size=2,
                input_ids=torch.tensor([10, 11], dtype=torch.int64, device="cuda"),
                positions=torch.tensor([20, 21], dtype=torch.int64, device="cuda"),
            )
            payload = recorder.end_forward(batch)
            torch.cuda.synchronize()

            expected_q, expected_scales = quantize_and_pack_int4(
                updated_hidden.cpu(), group_size=64
            )
            self.assertTrue(torch.equal(payload.activation_q.cpu(), expected_q))
            self.assertTrue(
                torch.equal(payload.activation_scales.cpu(), expected_scales)
            )
            self.assertTrue(torch.equal(payload.expert_ids.cpu(), updated_ids.cpu()))
            self.assertTrue(
                torch.equal(payload.expert_weights.cpu(), updated_weights.cpu())
            )
            self.assertTrue(torch.all(payload.site_valid).item())
            recorder.close()


if __name__ == "__main__":
    unittest.main()
