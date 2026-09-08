"""CPU contracts for the Qwen3.5 NVFP4 expert-pack artifact."""

import copy
import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.qwen3_5.expert_pack.build import (
    CheckpointExpertSource,
    _BuildLock,
    _check_static_layer_input_scale,
    _manifest_dict,
    encode_expert_payload,
)
from sglang.srt.layers.qwen3_5.expert_pack.format import (
    ALIGNMENT,
    COMPONENTS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPERT_LAYERS,
    EXPERTS_PER_LAYER,
    FORMAT_ID,
    PACK_SIZE,
    PAYLOAD_SIZE,
    RECORD_COUNT,
    RECORD_STRIDE,
    ExpertPackManifest,
    RecordInfo,
    record_index,
    record_key,
    record_offset,
)
from sglang.srt.layers.qwen3_5.expert_pack.validate import _validate_pack_bytes

MODEL_DIR = Path(
    os.environ.get(
        "QWEN35_MODEL_DIR",
        "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
    )
)


class _FakeSource:
    def __init__(self):
        self.values = {}
        prefix = "model.language_model.layers.1.mlp.experts.0."
        projections = {
            "gate_proj": (1, 3, 1.25, 2.5),
            "up_proj": (2, 4, 1.25, 2.5),
            "down_proj": (5, 6, 3.25, 4.5),
        }
        for projection, (
            data,
            scale,
            weight_global,
            input_global,
        ) in projections.items():
            rows, packed_k = (512, 1024) if projection != "down_proj" else (2048, 256)
            stem = prefix + projection + "."
            self.values[stem + "weight_packed"] = torch.full(
                (rows, packed_k), data, dtype=torch.uint8
            )
            self.values[stem + "weight_scale"] = torch.full(
                (rows, packed_k // 8), scale, dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            self.values[stem + "weight_global_scale"] = torch.tensor(
                [weight_global], dtype=torch.float32
            )
            self.values[stem + "input_global_scale"] = torch.tensor(
                [input_global], dtype=torch.float32
            )

    def get_tensor(self, name):
        return self.values[name]


def _records(first_sha256: str = "0" * 64) -> list[RecordInfo]:
    values = []
    for index in range(RECORD_COUNT):
        layer_id, expert_id = record_key(index)
        values.append(
            RecordInfo(
                layer_id,
                expert_id,
                index * RECORD_STRIDE,
                PAYLOAD_SIZE,
                first_sha256 if index == 0 else "0" * 64,
            )
        )
    return values


def _manifest_data(first_sha256: str = "0" * 64):
    return _manifest_dict(
        source={
            "model_dir": "/checkpoint",
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "index_sha256": EXPECTED_INDEX_SHA256,
            "shards": [{"file": "model-00001-of-00007.safetensors", "size": 1}],
        },
        records=_records(first_sha256),
        pack_sha256="0" * 64,
    )


class TestExpertPackFormat(unittest.TestCase):
    def test_fixed_layout_is_dense_aligned_and_layer_major(self):
        self.assertEqual(FORMAT_ID, "SGLANG-QWEN35-NVFP4-EXPERTPACK-v1")
        self.assertEqual(PAYLOAD_SIZE, 1_769_488)
        self.assertEqual(RECORD_STRIDE, 1_773_568)
        self.assertEqual(RECORD_STRIDE % ALIGNMENT, 0)
        self.assertEqual(RECORD_COUNT, 38 * 256)
        self.assertEqual(PACK_SIZE, 17_253_269_504)
        self.assertEqual(COMPONENTS["down.input_global_scale"].end, PAYLOAD_SIZE)
        self.assertEqual(record_index(1, 0), 0)
        self.assertEqual(record_index(38, 255), RECORD_COUNT - 1)
        self.assertEqual(record_key(RECORD_COUNT - 1), (38, 255))
        self.assertEqual(record_offset(2, 0), EXPERTS_PER_LAYER * RECORD_STRIDE)

    def test_manifest_requires_complete_exact_order_and_source_identity(self):
        data = _manifest_data()
        manifest = ExpertPackManifest.from_dict(data)
        self.assertEqual(manifest.record(1, 0).offset, 0)
        self.assertEqual(
            manifest.record(38, 255).offset, (RECORD_COUNT - 1) * RECORD_STRIDE
        )

        incomplete = dict(data, complete=False)
        with self.assertRaisesRegex(ValueError, "complete"):
            ExpertPackManifest.from_dict(incomplete)

        wrong_identity = copy.copy(data)
        wrong_identity["source"] = dict(data["source"], index_sha256="1" * 64)
        with self.assertRaisesRegex(ValueError, "index identity"):
            ExpertPackManifest.from_dict(wrong_identity)

        wrong_order = copy.copy(data)
        wrong_order["records"] = list(data["records"])
        wrong_order["records"][1] = dict(wrong_order["records"][1], expert_id=2)
        with self.assertRaisesRegex(ValueError, "record 1"):
            ExpertPackManifest.from_dict(wrong_order)


class TestExpertEncoding(unittest.TestCase):
    def test_payload_merges_gate_up_without_changing_source_bytes(self):
        payload = encode_expert_payload(_FakeSource(), 1, 0)
        self.assertEqual(len(payload), PAYLOAD_SIZE)
        gate_up = COMPONENTS["gate_up.data"]
        self.assertEqual(payload[gate_up.offset], 1)
        self.assertEqual(payload[gate_up.offset + gate_up.nbytes // 2], 2)
        gate_up_scale = COMPONENTS["gate_up.block_scale"]
        self.assertEqual(payload[gate_up_scale.offset], 3)
        self.assertEqual(payload[gate_up_scale.offset + gate_up_scale.nbytes // 2], 4)
        down = COMPONENTS["down.data"]
        self.assertEqual(payload[down.offset], 5)
        self.assertEqual(payload[COMPONENTS["down.block_scale"].offset], 6)
        self.assertEqual(
            struct.unpack_from(
                "<f", payload, COMPONENTS["gate_up.global_scale"].offset
            )[0],
            1.25,
        )

    def test_payload_rejects_gate_up_scale_mismatch(self):
        source = _FakeSource()
        name = (
            "model.language_model.layers.1.mlp.experts.0." "up_proj.weight_global_scale"
        )
        source.values[name] = torch.tensor([9.0], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "gate/up weight global scales differ"):
            encode_expert_payload(source, 1, 0)

    def test_builder_rejects_input_scale_change_between_experts(self):
        first = encode_expert_payload(_FakeSource(), 1, 0)
        second_source = _FakeSource()
        prefix = "model.language_model.layers.1.mlp.experts.0."
        for projection in ("gate_proj", "up_proj"):
            second_source.values[prefix + projection + ".input_global_scale"] = (
                torch.tensor([7.0], dtype=torch.float32)
            )
        second = encode_expert_payload(second_source, 1, 0)
        reference = _check_static_layer_input_scale(
            None, first, layer_id=1, expert_id=0
        )
        with self.assertRaisesRegex(ValueError, "static across all experts"):
            _check_static_layer_input_scale(reference, second, layer_id=1, expert_id=1)

    @unittest.skipUnless(MODEL_DIR.is_dir(), "Qwen3.5 checkpoint unavailable")
    def test_real_checkpoint_first_payload_golden_digest(self):
        with CheckpointExpertSource(MODEL_DIR) as source:
            payload = encode_expert_payload(source, 1, 0)
        self.assertEqual(len(payload), PAYLOAD_SIZE)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "489e8f4567ea5a4c94b8c32a51eb7b94544ca99ebac37196c6c444c0e7bdbebe",
        )


class TestPackValidation(unittest.TestCase):
    def test_builder_lock_rejects_concurrent_publisher(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "artifact"
            output_dir.mkdir()
            with _BuildLock(output_dir):
                with self.assertRaisesRegex(
                    RuntimeError, "another expert-pack builder"
                ):
                    with _BuildLock(output_dir):
                        self.fail("the second publisher must not acquire the lock")

    def test_selected_record_detects_payload_corruption_without_full_scan(self):
        payload_digest = hashlib.sha256(bytes(PAYLOAD_SIZE)).hexdigest()
        manifest = ExpertPackManifest.from_dict(_manifest_data(payload_digest))
        with tempfile.TemporaryDirectory() as temporary:
            pack_path = Path(temporary) / "experts.pack"
            with pack_path.open("wb") as handle:
                handle.truncate(PACK_SIZE)
            result = _validate_pack_bytes(
                manifest,
                pack_path,
                (0,),
                verify_pack_sha256=False,
                verify_padding=True,
            )
            self.assertEqual(result, (1, False, 1))
            with pack_path.open("r+b", buffering=0) as handle:
                handle.write(b"x")
                handle.flush()
            with self.assertRaisesRegex(ValueError, "payload SHA-256 mismatch"):
                _validate_pack_bytes(
                    manifest,
                    pack_path,
                    (0,),
                    verify_pack_sha256=False,
                    verify_padding=True,
                )


if __name__ == "__main__":
    unittest.main()
