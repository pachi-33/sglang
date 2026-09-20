import unittest
from types import SimpleNamespace

from sglang.srt.moe_trace.config import validate_moe_trace_server_args


def _server_args(**overrides):
    values = {
        "moe_trace_output_dir": "/tmp/moe-trace",
        "moe_trace_expert_routes": True,
        "moe_trace_router_inputs": False,
        "moe_trace_max_decode_tokens": 4,
        "moe_trace_activation_group_size": 128,
        "moe_trace_queue_depth": 2,
        "moe_trace_overflow_policy": "block",
        "device": "cuda",
        "speculative_algorithm": None,
        "dllm_algorithm": None,
        "pp_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestMoeTraceResolvedConfigValidation(unittest.TestCase):
    def test_cuda_non_speculative_non_dllm_is_allowed(self):
        validate_moe_trace_server_args(_server_args())

    def test_cpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA devices only"):
            validate_moe_trace_server_args(_server_args(device="cpu"))

    def test_speculative_decoding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "speculative decoding"):
            validate_moe_trace_server_args(_server_args(speculative_algorithm="EAGLE"))

    def test_dllm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "diffusion LLM"):
            validate_moe_trace_server_args(_server_args(dllm_algorithm="DDIM"))

    def test_pipeline_parallelism_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            validate_moe_trace_server_args(_server_args(pp_size=2))
