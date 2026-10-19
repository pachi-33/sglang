import ast
import unittest
from pathlib import Path

_ROOT = Path(__file__).parents[4]


def _function_source(path: Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(path.read_text(), item) or ""
    raise AssertionError(f"{class_name}.{method_name} was not found")


class TestRouterAdapters(unittest.TestCase):
    def test_public_and_custom_routers_have_trace_taps(self):
        topk = _ROOT / "python/sglang/srt/layers/moe/topk.py"
        hash_topk = _ROOT / "python/sglang/srt/layers/moe/hash_topk.py"
        inkling = _ROOT / "python/sglang/srt/models/inkling_common/moe.py"
        transformers = _ROOT / "python/sglang/srt/models/transformers.py"

        topk_source = topk.read_text()
        self.assertIn("capture_router_input(self, hidden_states)", topk_source)
        self.assertIn("capture_route(", topk_source)
        self.assertIn("trace_module=self", topk_source)
        self.assertIn(
            "capture_router_input(self, hidden_states)",
            _function_source(hash_topk, "HashTopK", "forward"),
        )
        self.assertIn(
            "capture_route(", _function_source(hash_topk, "HashTopK", "forward")
        )
        self.assertIn(
            "capture_router_input(self, x)",
            _function_source(inkling, "InklingGate", "forward"),
        )
        self.assertIn(
            "capture_route(self, topk_indices, routed_weights)",
            _function_source(inkling, "InklingGate", "forward"),
        )
        self.assertIn(
            "capture_route(self, topk_ids, topk_weights)",
            _function_source(transformers, "TransformersFusedMoE", "forward"),
        )
