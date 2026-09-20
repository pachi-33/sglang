"""CUDA regression tests for MoE trace int4 packing."""

import unittest

import torch

from sglang.srt.moe_trace.codec import quantize_and_pack_int4
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestInt4PackCuda(unittest.TestCase):
    def test_partial_group_does_not_write_past_packed_output(self):
        hidden, group_size = 131, 64
        packed_width = (hidden + 1) // 2
        num_groups = (hidden + group_size - 1) // group_size
        x = torch.linspace(-3.17, 4.23, hidden, device="cuda", dtype=torch.float32)[
            None
        ]
        expected_q, expected_scales = quantize_and_pack_int4(x.cpu(), group_size)

        # The output view has the required logical shape but extra storage at
        # the end of its row.  A tail-store mask that only checks group-local
        # offsets would overwrite this sentinel for the final partial group.
        q_backing = torch.full(
            (1, packed_width + 8), 0xA5, dtype=torch.uint8, device="cuda"
        )
        scale_backing = torch.full(
            (1, num_groups + 2), -1.0, dtype=torch.float16, device="cuda"
        )
        out_q = q_backing[:, :packed_width]
        out_scales = scale_backing[:, :num_groups]
        actual_q, actual_scales = quantize_and_pack_int4(
            x, group_size, out_q, out_scales
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(actual_q.cpu(), expected_q))
        self.assertTrue(torch.equal(actual_scales.cpu(), expected_scales))
        self.assertTrue(torch.all(q_backing[:, packed_width:] == 0xA5))
        self.assertTrue(torch.all(scale_backing[:, num_groups:] == -1.0))


if __name__ == "__main__":
    unittest.main()
