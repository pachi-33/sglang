"""CPU tests for the optional MoE trace int4 activation codec."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.moe_trace.codec import (
    dequantize_packed_int4,
    pack_int4,
    quantize_and_pack_int4,
    unpack_int4,
)


class TestInt4Codec(unittest.TestCase):
    def test_twos_complement_nibble_order(self):
        q = torch.tensor([[-8, -7, -1, 0, 1, 7, 3]], dtype=torch.int8)
        packed = pack_int4(q)
        self.assertEqual(packed.tolist(), [[0x98, 0x0F, 0x71, 0x03]])
        self.assertTrue(torch.equal(unpack_int4(packed, q.shape[1]), q))

    def test_zero_values_use_unit_scale(self):
        x = torch.zeros((2, 129), dtype=torch.float32)
        packed, scales = quantize_and_pack_int4(x)
        self.assertTrue(torch.equal(packed, torch.zeros_like(packed)))
        self.assertTrue(torch.equal(scales, torch.ones_like(scales)))
        self.assertTrue(
            torch.equal(dequantize_packed_int4(packed, scales, 129), x.half())
        )

    def test_signed_values_and_saturation(self):
        x = torch.tensor([[-100.0, -7.0, -1.0, 0.0, 1.0, 7.0, 100.0]])
        packed, scales = quantize_and_pack_int4(x, group_size=64)
        q = unpack_int4(packed, x.shape[1])
        self.assertEqual(q.tolist(), [[-7, 0, 0, 0, 0, 0, 7]])
        self.assertTrue(torch.all(scales > 0))

    def test_odd_group_size_is_rejected(self):
        x = torch.ones((1, 17), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "must be even"):
            quantize_and_pack_int4(x, group_size=63)

    def test_odd_hidden_and_non_multiple_group(self):
        x = torch.linspace(-3.0, 4.0, 131, dtype=torch.float32).reshape(1, -1)
        packed, scales = quantize_and_pack_int4(x, group_size=64)
        self.assertEqual(packed.shape, (1, 66))
        self.assertEqual(scales.shape, (1, 3))
        self.assertEqual((packed[0, -1] >> 4).item(), 0)
        decoded = dequantize_packed_int4(packed, scales, 131, group_size=64)
        bound = scales.float().repeat_interleave(64, dim=1)[:, :131] / 2 + 1e-3
        self.assertTrue(torch.all((decoded.float() - x).abs() <= bound))

    def test_group_sizes_dtypes_and_preallocated_outputs(self):
        values = torch.tensor(
            [[-2.5, -1.25, -0.1, 0.0, 0.1, 1.25, 2.5] * 19], dtype=torch.float32
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for group_size in (64, 128):
                x = values.to(dtype)
                packed_width = (x.shape[1] + 1) // 2
                groups = (x.shape[1] + group_size - 1) // group_size
                backing_q = torch.full((1, packed_width + 3), 255, dtype=torch.uint8)
                backing_s = torch.full((1, groups + 2), -1.0, dtype=torch.float16)
                out_q = backing_q[:, 1 : 1 + packed_width]
                out_s = backing_s[:, 1 : 1 + groups]
                packed, scales = quantize_and_pack_int4(x, group_size, out_q, out_s)
                self.assertEqual(packed.data_ptr(), out_q.data_ptr())
                self.assertEqual(scales.data_ptr(), out_s.data_ptr())
                decoded = dequantize_packed_int4(packed, scales, x.shape[1], group_size)
                bound = scales.float().repeat_interleave(group_size, dim=1)[
                    :, : x.shape[1]
                ]
                bound = bound / 2 + 2e-3
                self.assertTrue(torch.all((decoded.float() - x.float()).abs() <= bound))


if __name__ == "__main__":
    unittest.main()
