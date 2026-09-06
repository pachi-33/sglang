import unittest

import torch

from test.Qwen3_5MoECompat.reference.codec import (
    decode_e2m1,
    decode_e4m3fn,
    encode_e2m1,
    encode_e4m3fn,
    quantize_a4,
    unpack_a4,
)


class TestReferenceCodec(unittest.TestCase):
    def test_signed_zero_and_nan_contract(self):
        values = torch.tensor([-0.0, 0.0, -1e-9, 1e-9, -448.0, 448.0])
        decoded = decode_e4m3fn(encode_e4m3fn(values))
        self.assertTrue(torch.signbit(decoded[0]))
        self.assertFalse(torch.signbit(decoded[1]))
        self.assertTrue(torch.isnan(decode_e4m3fn(torch.tensor([127, 255], dtype=torch.uint8))).all())
        decoded_e2 = decode_e2m1(encode_e2m1(values))
        self.assertTrue(torch.signbit(decoded_e2[0]))
        self.assertFalse(torch.signbit(decoded_e2[1]))

    def test_a4_zero_scale_uses_zero_codes(self):
        packed, scale, global_scale = quantize_a4(torch.zeros((2, 16)), torch.tensor([7.0]))
        self.assertTrue(torch.equal(packed, torch.zeros_like(packed)))
        self.assertTrue(torch.equal(scale, torch.zeros_like(scale)))
        self.assertTrue(torch.equal(unpack_a4(packed, scale, global_scale), torch.zeros((2, 16))))
