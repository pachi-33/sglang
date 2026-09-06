import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.hf_transformers_utils import get_config
from sglang.srt.layers.qwen3_5.config import Qwen3_5MoeConfig


class TestQwen35Config(unittest.TestCase):
    def test_nested_text_config_loads_before_auto_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "model_type": "qwen3_5_moe",
                "language_model_only": True,
                "text_config": {"model_type": "qwen3_5_moe_text", "vocab_size": 248320, "hidden_size": 2048},
            }
            Path(directory, "config.json").write_text(json.dumps(config))
            loaded = get_config(directory, trust_remote_code=False)
        self.assertIsInstance(loaded, Qwen3_5MoeConfig)
        self.assertEqual((loaded.text_config.vocab_size, loaded.text_config.hidden_size), (248320, 2048))

