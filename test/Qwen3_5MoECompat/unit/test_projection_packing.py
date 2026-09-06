"""CPU contracts for byte-preserving Qwen3.5 projection packing."""

import unittest

import torch

from sglang.srt.layers.qwen3_5.checkpoint import Qwen35Checkpoint


def _fp8_bytes(rows: int, marker: int) -> torch.Tensor:
    """Create raw E4M3 storage without evaluating its deliberately arbitrary bytes."""
    return torch.full((rows, 2048), marker, dtype=torch.uint8).view(torch.float8_e4m3fn)


def _scale(rows: int, marker: int) -> torch.Tensor:
    return torch.full((rows // 128, 16), marker, dtype=torch.float16)


class TestMergedProjectionLayout(unittest.TestCase):
    def test_gdn_helper_packs_qkv_z_and_ba_with_aliases(self):
        qkv, z = _fp8_bytes(8192, 1), _fp8_bytes(4096, 2)
        qsf, zsf = _scale(8192, 3), _scale(4096, 4)
        b = torch.full((32, 2048), 5, dtype=torch.float16)
        a = torch.full((32, 2048), 6, dtype=torch.float16)
        raw = {
            "linear_attn.in_proj_qkv.weight": qkv,
            "linear_attn.in_proj_qkv.weight_scale": qsf,
            "linear_attn.in_proj_z.weight": z,
            "linear_attn.in_proj_z.weight_scale": zsf,
            "linear_attn.in_proj_b.weight": b,
            "linear_attn.in_proj_a.weight": a,
        }

        packed = Qwen35Checkpoint._pack_gdn_projections(raw)
        self.assertFalse(raw)
        merged = packed["linear_attn.in_proj_qkv_z"]
        ba = packed["linear_attn.in_proj_ba"]
        self.assertEqual(merged.logical_shape, (12288, 2048))
        self.assertTrue(
            torch.equal(
                merged.data, torch.cat((qkv.view(torch.uint8), z.view(torch.uint8)))
            )
        )
        self.assertTrue(torch.equal(merged.block_scale, torch.cat((qsf, zsf))))
        for name, expected_data, expected_scale in (
            ("linear_attn.in_proj_qkv", qkv.view(torch.uint8), qsf),
            ("linear_attn.in_proj_z", z.view(torch.uint8), zsf),
        ):
            component = packed[name]
            self.assertEqual(
                component.data.untyped_storage().data_ptr(),
                merged.data.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                component.block_scale.untyped_storage().data_ptr(),
                merged.block_scale.untyped_storage().data_ptr(),
            )
            self.assertTrue(torch.equal(component.data, expected_data))
            self.assertTrue(torch.equal(component.block_scale, expected_scale))
        self.assertTrue(torch.equal(ba.data, torch.cat((b, a))))
        for name, expected in (
            ("linear_attn.in_proj_b", b),
            ("linear_attn.in_proj_a", a),
        ):
            component = packed[name]
            self.assertEqual(
                component.data.untyped_storage().data_ptr(),
                ba.data.untyped_storage().data_ptr(),
            )
            self.assertTrue(torch.equal(component.data, expected))

    def test_full_helper_packs_qgate_k_v_with_aliases(self):
        qgate, key, value = _fp8_bytes(8192, 1), _fp8_bytes(512, 2), _fp8_bytes(512, 3)
        qsf, ksf, vsf = _scale(8192, 4), _scale(512, 5), _scale(512, 6)
        raw = {
            "self_attn.q_proj.weight": qgate,
            "self_attn.q_proj.weight_scale": qsf,
            "self_attn.k_proj.weight": key,
            "self_attn.k_proj.weight_scale": ksf,
            "self_attn.v_proj.weight": value,
            "self_attn.v_proj.weight_scale": vsf,
        }

        packed = Qwen35Checkpoint._pack_full_projections(raw)
        self.assertFalse(raw)
        merged = packed["self_attn.qkv_proj"]
        self.assertEqual(merged.logical_shape, (9216, 2048))
        self.assertTrue(
            torch.equal(
                merged.data,
                torch.cat(
                    (
                        qgate.view(torch.uint8),
                        key.view(torch.uint8),
                        value.view(torch.uint8),
                    )
                ),
            )
        )
        self.assertTrue(torch.equal(merged.block_scale, torch.cat((qsf, ksf, vsf))))
        for name, expected_data, expected_scale in (
            ("self_attn.q_proj", qgate.view(torch.uint8), qsf),
            ("self_attn.k_proj", key.view(torch.uint8), ksf),
            ("self_attn.v_proj", value.view(torch.uint8), vsf),
        ):
            component = packed[name]
            self.assertEqual(
                component.data.untyped_storage().data_ptr(),
                merged.data.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                component.block_scale.untyped_storage().data_ptr(),
                merged.block_scale.untyped_storage().data_ptr(),
            )
            self.assertTrue(torch.equal(component.data, expected_data))
            self.assertTrue(torch.equal(component.block_scale, expected_scale))


class TestProjectionDeviceAliases(unittest.TestCase):
    """The load path transfers only merged projection storage to CUDA."""

    @unittest.skipUnless(
        torch.cuda.is_available()
        and __import__("pathlib")
        .Path(
            __import__("os").environ.get(
                "QWEN35_MODEL_DIR",
                "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
            )
        )
        .is_dir(),
        "real Qwen3.5 checkpoint/CUDA unavailable",
    )
    def test_real_layer_projection_aliases_have_no_extra_device_storage(self):
        import os
        from pathlib import Path

        loader = Qwen35Checkpoint(
            Path(
                os.environ.get(
                    "QWEN35_MODEL_DIR",
                    "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
                )
            )
        )
        for layer, merged_key, aliases in (
            (
                0,
                "linear_attn.in_proj_qkv_z",
                ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
            ),
            (
                3,
                "self_attn.qkv_proj",
                ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            ),
        ):
            weights = loader.load_layer(layer, "cuda")
            merged = weights[merged_key]
            self.assertTrue(merged.data.is_cuda)
            for key in aliases:
                component = weights[key]
                self.assertEqual(
                    component.data.untyped_storage().data_ptr(),
                    merged.data.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    component.block_scale.untyped_storage().data_ptr(),
                    merged.block_scale.untyped_storage().data_ptr(),
                )
            if layer == 0:
                ba = weights["linear_attn.in_proj_ba"]
                for key in ("linear_attn.in_proj_b", "linear_attn.in_proj_a"):
                    self.assertEqual(
                        weights[key].data.untyped_storage().data_ptr(),
                        ba.data.untyped_storage().data_ptr(),
                    )
                merged_payloads = (merged.data, merged.block_scale, ba.data)
                component_payloads = tuple(
                    weights[key].data
                    for key in (
                        *aliases,
                        "linear_attn.in_proj_b",
                        "linear_attn.in_proj_a",
                    )
                ) + tuple(weights[key].block_scale for key in aliases)
            else:
                merged_payloads = (merged.data, merged.block_scale)
                component_payloads = tuple(
                    weights[key].data for key in aliases
                ) + tuple(weights[key].block_scale for key in aliases)
            unique_bytes = {
                payload.untyped_storage().data_ptr(): payload.untyped_storage().nbytes()
                for payload in (*merged_payloads, *component_payloads)
            }
            self.assertEqual(
                sum(unique_bytes.values()),
                sum(payload.untyped_storage().nbytes() for payload in merged_payloads),
            )
            del weights
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
