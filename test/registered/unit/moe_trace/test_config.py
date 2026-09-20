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
        "enable_dp_attention": False,
        "tp_size": 1,
        "dp_size": 1,
        "moe_dp_size": 1,
        "dwdp_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestMoeTraceResolvedConfigValidation(unittest.TestCase):
    def test_cuda_route_only_is_allowed(self):
        validate_moe_trace_server_args(_server_args())

    def test_cuda_router_inputs_only_is_allowed(self):
        validate_moe_trace_server_args(
            _server_args(moe_trace_expert_routes=False, moe_trace_router_inputs=True)
        )

    def test_cuda_routes_and_router_inputs_are_allowed(self):
        validate_moe_trace_server_args(_server_args(moe_trace_router_inputs=True))

    def test_cuda_parallelism_is_unchanged(self):
        validate_moe_trace_server_args(
            _server_args(
                enable_dp_attention=True,
                dp_size=2,
                moe_dp_size=2,
                dwdp_size=2,
            )
        )

    def test_npu_route_only_is_allowed(self):
        validate_moe_trace_server_args(_server_args(device="npu"))

    def test_npu_router_inputs_only_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "router-inputs is CUDA-only"):
            validate_moe_trace_server_args(
                _server_args(
                    device="npu",
                    moe_trace_expert_routes=False,
                    moe_trace_router_inputs=True,
                )
            )

    def test_npu_routes_and_router_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "router-inputs is CUDA-only"):
            validate_moe_trace_server_args(
                _server_args(device="npu", moe_trace_router_inputs=True)
            )

    def test_npu_dp_attention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tensor parallelism only"):
            validate_moe_trace_server_args(
                _server_args(device="npu", enable_dp_attention=True)
            )

    def test_npu_data_parallel_dimensions_are_rejected(self):
        for field in ("dp_size", "moe_dp_size", "dwdp_size"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "tensor parallelism only"),
            ):
                validate_moe_trace_server_args(_server_args(device="npu", **{field: 2}))

    def test_npu_tp_greater_than_one_is_allowed(self):
        validate_moe_trace_server_args(_server_args(device="npu", tp_size=2))

    def test_cpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA and NPU route-only"):
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
