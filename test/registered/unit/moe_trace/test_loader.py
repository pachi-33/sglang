import json
import tempfile
import unittest

import torch

from sglang.srt.moe_trace.loader import MoeTraceLoader
from sglang.srt.moe_trace.types import (
    MoeTraceBatchOutput,
    MoeTraceSite,
    MoeTraceSiteLayout,
)
from sglang.srt.moe_trace.writer import MoeTraceWriter, safe_request_id


def _layout():
    return [
        MoeTraceSiteLayout(MoeTraceSite(1, "moe", 0, "topk", 4, 8, 2), 0, 4, 0, 1, 0)
    ]


def _payload(ids, offset=0):
    n = len(ids)
    return MoeTraceBatchOutput(
        request_ids=ids,
        input_token_ids=torch.arange(offset, offset + n),
        positions=torch.arange(offset, offset + n),
        site_valid=torch.ones(n, 1, dtype=torch.bool),
        activation_q=torch.ones(n, 4, dtype=torch.uint8),
        activation_scales=torch.ones(n, 1, dtype=torch.float16),
        expert_ids=torch.ones(n, 2, dtype=torch.int32),
        expert_weights=torch.ones(n, 2, dtype=torch.float16),
    )


class TestMoeTraceLoader(unittest.TestCase):
    def test_loader_round_trip_multiple_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = MoeTraceWriter(temp, _layout())
            writer.submit(_payload(["request"], 5))
            writer.submit(_payload(["request"], 6))
            writer.finalize_request("request")
            writer.close()
            loader = MoeTraceLoader(temp)
            chunks = list(loader.iter_chunks("request"))
            loaded = loader.load_trace("request")
            self.assertEqual(len(chunks), 2)
            self.assertEqual(loaded["input_token_ids"].tolist(), [5, 6])
            self.assertEqual(loaded["site.1.activation_q"].shape, (2, 4))
            self.assertEqual(loader.manifest("request")["status"], "complete")

    def test_loader_rejects_bad_hash(self):
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            writer = MoeTraceWriter(root, _layout())
            writer.submit(_payload(["request"]))
            writer.close()
            manifest_path = root / safe_request_id("request") / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["chunks"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                list(MoeTraceLoader(root).iter_chunks("request"))
