from test.Qwen3_5MoECompat.reference.codec import decode_e2m1 as ref_decode_e2m1
from test.Qwen3_5MoECompat.reference.codec import decode_e4m3fn as ref_decode_e4m3fn
from test.Qwen3_5MoECompat.reference.codec import encode_e2m1 as ref_encode_e2m1
from test.Qwen3_5MoECompat.reference.codec import encode_e4m3fn as ref_encode_e4m3fn
from test.Qwen3_5MoECompat.reference.codec import quantize_a4, quantize_a8
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase

import torch

from sglang.srt.layers.qwen3_5.quantization import (
    decode_e2m1,
    decode_e4m3fn,
    encode_e2m1,
    encode_e4m3fn,
    linear_fp8,
    pack_e2m1,
    quantize_fp8,
    quantize_nvfp4,
    unpack_e2m1,
)
from sglang.srt.layers.qwen3_5.weights import Weight


class TestQwen35Quantization(V100TestCase):
    def test_e2m1_rne_and_signed_zero(self):
        x = torch.tensor([-0.0, 0.0, 0.25, 0.75, 1.25, 1.75, 5.0], device="cuda")
        # Ties select the even representable E2M1 code.
        self.assertEqual(encode_e2m1(x).tolist(), [8, 0, 0, 2, 2, 4, 6])
        codes = torch.arange(16, dtype=torch.uint8, device="cuda")
        self.assertTrue(torch.equal(encode_e2m1(x), ref_encode_e2m1(x)))
        self.assertTrue(torch.equal(decode_e2m1(codes), ref_decode_e2m1(codes)))
        self.assertTrue(torch.equal(encode_e2m1(ref_decode_e2m1(codes)), codes))
        values = ref_decode_e2m1(torch.arange(8, dtype=torch.uint8, device="cuda"))
        midpoint = (values[:-1] + values[1:]) * 0.5
        below = torch.nextafter(midpoint, torch.full_like(midpoint, -float("inf")))
        above = torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))
        boundary_inputs = torch.cat((below, midpoint, above, -below, -midpoint, -above))
        self.assertTrue(
            torch.equal(encode_e2m1(boundary_inputs), ref_encode_e2m1(boundary_inputs))
        )

    def test_e4m3_subnormal_carry_to_normal(self):
        # 7.75 / 512 rounds to 8 / 512 == the smallest normal 2^-6.
        x = torch.tensor([0.0, 2.0**-9, 7.75 / 512.0, 2.0**-6, 448.0], device="cuda")
        codes = encode_e4m3fn(x)
        self.assertEqual(codes.tolist(), [0, 1, 8, 8, 126])
        all_codes = torch.arange(256, dtype=torch.uint8, device="cuda")
        valid = (
            torch.cat((torch.arange(127), torch.arange(128, 255)))
            .to(torch.uint8)
            .cuda()
        )
        self.assertTrue(torch.equal(encode_e4m3fn(x), ref_encode_e4m3fn(x)))
        torch.testing.assert_close(
            decode_e4m3fn(all_codes), ref_decode_e4m3fn(all_codes), equal_nan=True
        )
        self.assertTrue(torch.equal(encode_e4m3fn(ref_decode_e4m3fn(valid)), valid))
        positive = ref_decode_e4m3fn(
            torch.arange(127, dtype=torch.uint8, device="cuda")
        )
        midpoint = (positive[:-1] + positive[1:]) * 0.5
        below = torch.nextafter(midpoint, torch.full_like(midpoint, -float("inf")))
        above = torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))
        boundary_inputs = torch.cat((below, midpoint, above, -below, -midpoint, -above))
        self.assertTrue(
            torch.equal(
                encode_e4m3fn(boundary_inputs), ref_encode_e4m3fn(boundary_inputs)
            )
        )

    def test_nvfp4_static_global_and_nibble_order(self):
        x = torch.tensor(
            [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 64],
            dtype=torch.float16,
            device="cuda",
        )
        global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        q = quantize_nvfp4(x, global_scale)
        self.assertEqual(q.data[0, :4].tolist(), [0x10, 0x32, 0x54, 0x76])
        self.assertTrue(torch.equal(unpack_e2m1(q.data), encode_e2m1(x)))
        self.assertTrue(torch.equal(decode_e2m1(unpack_e2m1(q.data)), x.float()))
        self.assertEqual(pack_e2m1(unpack_e2m1(q.data)).tolist(), q.data.tolist())
        expected_data, expected_sf, _ = quantize_a4(x, global_scale)
        self.assertTrue(torch.equal(q.data, expected_data))
        self.assertTrue(torch.equal(q.block_scale, expected_sf))

    def test_nvfp4_zero_local_scale_has_zero_codes(self):
        # The standalone E2M1 codec preserves signed zero, but a zero NVFP4
        # local scale always carries zero E2M1 nibbles.
        x = torch.full((1, 16), -0.0, dtype=torch.float16, device="cuda")
        q = quantize_nvfp4(x, torch.tensor([1.0], dtype=torch.float32, device="cuda"))
        self.assertEqual(q.block_scale.tolist(), [[0]])
        self.assertEqual(q.data.tolist(), [[0] * 8])

    def test_fp8_zero_local_scale_preserves_signed_zero_codes(self):
        x = torch.tensor([[-0.0, 0.0] * 64], dtype=torch.float16, device="cuda")
        q = quantize_fp8(x)
        expected_data, expected_scale = quantize_a8(x)
        self.assertEqual(q.block_scale.tolist(), [[0.0]])
        self.assertEqual(q.data[0, :4].tolist(), [128, 0, 128, 0])
        self.assertTrue(torch.equal(q.data, expected_data))
        self.assertTrue(torch.equal(q.block_scale, expected_scale))

    def test_fp8_quantization_and_block_gemm(self):
        torch.manual_seed(11)
        for m, k in ((1, 2048), (17, 4096), (128, 2048)):
            with self.subTest(m=m, k=k):
                self._check_fp8_block_gemm(m, k)

        empty = quantize_fp8(torch.empty((0, 2048), dtype=torch.float16, device="cuda"))
        self.assertEqual(tuple(empty.data.shape), (0, 2048))
        weight = Weight(
            "fp8",
            torch.zeros((256, 2048), dtype=torch.uint8, device="cuda"),
            (256, 2048),
            torch.ones((2, 16), dtype=torch.float16, device="cuda"),
        )
        self.assertEqual(tuple(linear_fp8(empty, weight).shape), (0, 256))

    def _check_fp8_block_gemm(self, m: int, k: int) -> None:
        x = torch.randn((m, k), dtype=torch.float16, device="cuda")
        qx = quantize_fp8(x)
        expected_q, expected_scale = quantize_a8(x)
        self.assertTrue(torch.equal(qx.block_scale, expected_scale))
        self.assertTrue(torch.equal(qx.data, expected_q))

        n = 256
        groups = k // 128
        w_float = torch.randn((n, k), dtype=torch.float16, device="cuda") * 0.1
        _, per_row_scale = quantize_a8(w_float)
        w_scale = per_row_scale.reshape(2, 128, groups).amax(dim=1)
        w_raw = ref_encode_e4m3fn(
            (
                w_float.float().reshape(2, 128, groups, 128) / w_scale[:, None, :, None]
            ).reshape_as(w_float)
        )
        weight = Weight("fp8", w_raw, (n, k), w_scale.to(torch.float16))
        actual = linear_fp8(qx, weight)
        a = ref_decode_e4m3fn(qx.data).reshape(m, groups, 128).half()
        b = ref_decode_e4m3fn(w_raw).reshape(n, groups, 128).half()
        partial = torch.einsum("mgk,ngk->mng", a.float(), b.float())
        stored_w_scale = weight.block_scale.float().repeat_interleave(128, dim=0)
        expected = (
            partial * qx.block_scale[:, None, :] * stored_w_scale[None, :, :]
        ).sum(dim=-1)
        nrmse = torch.linalg.vector_norm(
            actual.float() - expected
        ) / torch.linalg.vector_norm(expected)
        value = float(nrmse)
        print(f"W8A8 M={m} N={n} K={k} NRMSE={value:.8g}")
        self.assertLess(value, 2e-3)


if __name__ == "__main__":
    import unittest

    unittest.main()
