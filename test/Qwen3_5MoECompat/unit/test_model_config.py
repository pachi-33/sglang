import json
import os
import tempfile
import unittest
from pathlib import Path

from sglang.srt.hf_transformers_utils import get_config
from sglang.srt.layers.qwen3_5.checkpoint import Qwen35Checkpoint
from sglang.srt.layers.qwen3_5.config import Qwen3_5MoeConfig


class TestQwen35Config(unittest.TestCase):
    def test_nested_text_config_loads_before_auto_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "model_type": "qwen3_5_moe",
                "language_model_only": True,
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "vocab_size": 248320,
                    "hidden_size": 2048,
                },
            }
            Path(directory, "config.json").write_text(json.dumps(config))
            loaded = get_config(directory, trust_remote_code=False)
        self.assertIsInstance(loaded, Qwen3_5MoeConfig)
        self.assertEqual(
            (loaded.text_config.vocab_size, loaded.text_config.hidden_size),
            (248320, 2048),
        )

    def test_rejects_incompatible_gdn_and_rope_contract(self):
        source = Path(
            os.environ.get(
                "QWEN35_MODEL_DIR",
                "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
            )
        )
        if not source.is_dir():
            self.skipTest("real checkpoint unavailable")
        config = json.loads((source / "config.json").read_text())
        for key, value in (("linear_conv_kernel_dim", 3), ("attn_output_gate", False)):
            altered = json.loads(json.dumps(config))
            altered["text_config"][key] = value
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    Qwen35Checkpoint.validate_config(directory)
        altered = json.loads(json.dumps(config))
        altered["text_config"]["rope_parameters"]["partial_rotary_factor"] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                Qwen35Checkpoint.validate_config(directory)
