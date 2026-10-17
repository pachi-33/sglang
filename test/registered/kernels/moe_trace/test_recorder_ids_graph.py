"""Real accelerator graph replay coverage for multi-request target ID tracing."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from sglang.srt.moe_trace.config import MoeTraceConfig
from sglang.srt.moe_trace.loader import MoeTraceBatchLoader
from sglang.srt.moe_trace.recorder import MoeTraceRecorder
from sglang.srt.moe_trace.types import MoeTraceSite


class _VerifyMode:
    def is_decode(self):
        return False

    def is_target_verify(self):
        return True


class TestMoeTraceIdsGraph(unittest.TestCase):
    def _run_replay(self, device):
        device_module = torch.get_device_module(device)
        with tempfile.TemporaryDirectory() as root:
            config = MoeTraceConfig(
                Path(root), True, False, 0, 128, 2, "block", ids_only=True
            )
            with patch(
                "sglang.srt.moe_trace.recorder.get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ):
                recorder = MoeTraceRecorder(
                    config,
                    [MoeTraceSite(0, "moe.topk", 0, "topk", 4, 32, 2)],
                    max_rows=8,
                    device=device,
                )
            try:
                module = SimpleNamespace(_moe_trace_site_id=0)
                ids = torch.zeros((8, 2), dtype=torch.int32, device=device)
                weights = torch.ones((8, 2), dtype=torch.float32, device=device)
                with recorder.capture_scope():
                    recorder.capture_route(module, ids, weights)
                device_module.synchronize()
                graph = (
                    device_module.CUDAGraph()
                    if device == "cuda"
                    else device_module.NPUGraph()
                )
                graph_options = (
                    {"auto_dispatch_capture": True} if device == "npu" else {}
                )
                with (
                    recorder.capture_scope(),
                    device_module.graph(graph, **graph_options),
                ):
                    recorder.capture_route(module, ids, weights)
                for value, requests in ((7, ["A", "B"]), (19, ["B", "A"])):
                    ids.fill_(value)
                    batch = SimpleNamespace(
                        forward_mode=_VerifyMode(),
                        rids=requests,
                        batch_size=2,
                        spec_info=SimpleNamespace(num_tokens_per_req=3),
                        input_ids=torch.arange(6, device=device),
                        positions=torch.arange(6, device=device),
                    )
                    context = recorder.prepare_forward(batch)
                    recorder.begin_forward(True)
                    graph.replay()
                    payload = recorder.end_forward(batch, context=context)
                    payload.map_device_tensors(lambda tensor: tensor.cpu())
                    recorder.submit(payload)
            finally:
                recorder.close()
            trace = MoeTraceBatchLoader(root).load_trace("A")
            self.assertEqual(
                trace["site.0.expert_ids"][:, 0].tolist(), [7] * 3 + [19] * 3
            )
            self.assertEqual(trace["candidate_indices"].tolist(), [0, 1, 2] * 2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_multi_request_verify_graph(self):
        self._run_replay("cuda")

    @unittest.skipUnless(
        hasattr(torch, "npu") and torch.npu.is_available(), "requires Ascend NPU"
    )
    def test_npu_multi_request_verify_graph(self):
        self._run_replay("npu")
