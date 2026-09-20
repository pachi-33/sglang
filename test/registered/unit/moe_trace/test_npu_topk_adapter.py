import ast
import unittest
from pathlib import Path

_ROOT = Path(__file__).parents[4]
_TOPK_PATH = _ROOT / "python/sglang/srt/layers/moe/topk.py"
_NPU_TOPK_PATH = _ROOT / "python/sglang/srt/hardware_backend/npu/moe/topk.py"


def _function(
    tree: ast.AST, name: str, class_name: str | None = None
) -> ast.FunctionDef:
    nodes = (
        tree.body
        if class_name is None
        else next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    )
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"{class_name + '.' if class_name else ''}{name} was not found"
    )


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"{name} keyword was not found")


class TestNpuTopKTraceAdapter(unittest.TestCase):
    """Source contracts: this module must stay import-free on CPU-only hosts."""

    def test_forward_npu_does_not_capture_router_inputs(self):
        forward_npu = _function(
            ast.parse(_TOPK_PATH.read_text()), "forward_npu", "TopK"
        )
        self.assertEqual(_calls(forward_npu, "_capture_router_input"), [])

    def test_forward_npu_passes_its_trace_module_to_fused_adapter(self):
        forward_npu = _function(
            ast.parse(_TOPK_PATH.read_text()), "forward_npu", "TopK"
        )
        fused_call = _calls(forward_npu, "fused_topk_npu")
        self.assertEqual(len(fused_call), 1)
        trace_module = _keyword(fused_call[0], "trace_module")
        self.assertIsInstance(trace_module, ast.Name)
        self.assertEqual(trace_module.id, "self")

    def test_fallback_forwards_trace_module_to_generic_post_process_tap(self):
        fused_npu = _function(ast.parse(_NPU_TOPK_PATH.read_text()), "fused_topk_npu")
        fallback_call = _calls(fused_npu, "select_experts")
        self.assertEqual(len(fallback_call), 1)
        trace_module = _keyword(fallback_call[0], "trace_module")
        self.assertIsInstance(trace_module, ast.Name)
        self.assertEqual(trace_module.id, "trace_module")

    def test_npu_capture_slices_shared_slots_before_logical_to_physical(self):
        source = _NPU_TOPK_PATH.read_text()
        fused_npu = _function(ast.parse(source), "fused_topk_npu")
        capture_call = _calls(fused_npu, "capture_route")
        mapping_call = _calls(fused_npu, "topk_ids_logical_to_physical")

        self.assertEqual(len(capture_call), 1)
        self.assertEqual(len(mapping_call), 1)
        self.assertLess(capture_call[0].lineno, mapping_call[0].lineno)

        capture_source = ast.get_source_segment(source, capture_call[0]) or ""
        self.assertIn("topk_ids[:, :routed_width]", capture_source)
        self.assertIn("topk_weights[:, :routed_width]", capture_source)
        self.assertIn(
            "routed_width = topk_ids.shape[-1] - topk_config.num_fused_shared_experts",
            source,
        )


if __name__ == "__main__":
    unittest.main()
