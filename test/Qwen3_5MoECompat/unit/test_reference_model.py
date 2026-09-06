"""CPU contracts for the independent FP8 block-semantic reference."""

import unittest
from test.Qwen3_5MoECompat.reference.codec import encode_e4m3fn
from test.Qwen3_5MoECompat.reference.model import (
    fp8_linear_payload,
    fp8_linear_payload_math,
)

import torch

from sglang.srt.layers.qwen3_5.weights import Weight


def _weight(rows=128, columns=256, scales=(3.0, 5.0)):
    data = encode_e4m3fn(torch.ones((rows, columns), dtype=torch.float32))
    block_scale = torch.tensor([list(scales)], dtype=torch.float16)
    return Weight("fp8", data, (rows, columns), block_scale)


class TestFp8BlockSemanticReference(unittest.TestCase):
    def test_k128_partials_apply_per_row_activation_then_weight_scales(self):
        payload = encode_e4m3fn(torch.ones((2, 256), dtype=torch.float32))
        activation_scale = torch.tensor([[2.0, 4.0], [1.0, 3.0]], dtype=torch.float32)
        weight = _weight()

        observed = fp8_linear_payload(payload, activation_scale, weight)
        # Each row has two K=128 partials.  A single post-GEMM scale would
        # fail this fixture because both activation and weight scales vary by K.
        expected = torch.tensor(
            [[128 * 2 * 3 + 128 * 4 * 5], [128 * 1 * 3 + 128 * 3 * 5]],
            dtype=torch.float16,
        )
        self.assertTrue(torch.equal(observed[:, :1], expected))
        self.assertTrue(torch.equal(observed, expected.expand_as(observed)))
        wrong_post_gemm_scale = torch.full_like(observed, (128 + 128) * 2 * 3)
        self.assertFalse(torch.equal(observed, wrong_post_gemm_scale))
        self.assertTrue(
            torch.equal(
                observed, fp8_linear_payload_math(payload, activation_scale, weight)
            )
        )

    def test_exact_scale_shapes_are_required(self):
        payload = encode_e4m3fn(torch.ones((2, 256), dtype=torch.float32))
        weight = _weight()
        for bad in (
            torch.ones((2, 1), dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.float32),
            torch.ones((2, 2), dtype=torch.float16),
        ):
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype):
                with self.assertRaisesRegex(ValueError, "activation scale"):
                    fp8_linear_payload(payload, bad, weight)

    def test_weight_k_and_scale_shape_errors_are_rejected(self):
        payload = encode_e4m3fn(torch.ones((1, 256), dtype=torch.float32))
        good_scale = torch.ones((1, 2), dtype=torch.float32)
        bad_k = _weight(columns=128, scales=(1.0,))
        with self.assertRaisesRegex(ValueError, "matching nonempty K"):
            fp8_linear_payload(payload, good_scale, bad_k)


if __name__ == "__main__":
    unittest.main()
