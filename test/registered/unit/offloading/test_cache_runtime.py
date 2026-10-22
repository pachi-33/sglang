"""Framework entry points against the single native runtime, with CPU device doubles."""

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from test_npu_backend import Assembly

from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def source_method(path, cls, name, **namespace):
    """Execute the real framework method without importing unrelated device packages."""
    path = Path(__file__).resolve().parents[4] / path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def batch(decode=False):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode_or_idle=lambda: decode,
            is_extend_without_speculative=lambda: not decode,
        ),
        input_ids=torch.ones(2, dtype=torch.int64),
        original_global_num_tokens_cpu=[2, 2],
        moe_num_token_non_padded=lambda: torch.tensor(2, dtype=torch.int32),
    )


class Decoder(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.mlp = SimpleNamespace(
            layer_id=layer_id,
            is_deepseek_v4=True,
            experts=SimpleNamespace(layer_id=layer_id),
        )
        self.calls = []

    def forward_hc_pre_from_prev(self, hidden):
        self.calls.append("body")
        return hidden

    def forward(self, hidden):
        return self.forward_hc_pre_from_prev(hidden)


class TestNativeCacheRuntime(unittest.TestCase):
    def setUp(self):
        self.assembly = Assembly(layer_count=1, local_experts=4)
        self.runtime = self.assembly.finalize()
        self.owner = MoEOffloadOwner(
            self.assembly.config, self.assembly.catalog, self.runtime
        )
        self.addCleanup(self.owner.close)
        self.decoder = Decoder(3)
        self.runner = SimpleNamespace(
            model=self.decoder, capture_head_hooks=[], capture_tail_hooks=[]
        )
        self.offloader = SimpleNamespace(owner=self.owner)

    def install(self):
        self.runtime.install_model_hooks(self.offloader, self.runner)

    def test_layer_hook_and_direct_entry_start_prefill_once(self):
        self.install()
        prefill = SimpleNamespace(
            layer_start=lambda layer, ctx: self.decoder.calls.append("start")
        )
        self.runtime.layer_programs[3].prefill = prefill
        for direct in (False, True):
            self.decoder.calls.clear()
            with self.runtime.forward_scope(self.owner, batch()):
                x = torch.ones(2, 2)
                result = (
                    self.decoder.forward_hc_pre_from_prev(x)
                    if direct
                    else self.decoder(x)
                )
                torch.testing.assert_close(result, x)
            self.assertEqual(self.decoder.calls, ["start", "body"])

    def test_fused_moe_entry_uses_current_batch_and_native_program(self):
        self.install()
        calls = []
        self.runtime.layer_programs[3].prefill = SimpleNamespace(
            layer_start=lambda *args: None,
            run=lambda layer, x, routing, ctx: (
                calls.append((layer, routing, ctx.num_token_non_padded)) or x * 2
            ),
        )
        forward = source_method(
            "python/sglang/srt/layers/moe/fused_moe_triton/layer.py",
            "FusedMoE",
            "forward_impl",
        )
        b = batch()
        with self.runtime.forward_scope(self.owner, b):
            # Eager preprocessing may rebind the token count after scope entry.
            live_count = torch.tensor(1, dtype=torch.int32)
            b.moe_num_token_non_padded = lambda: live_count
            result = forward(self.decoder.mlp.experts, torch.ones(2, 2), "routing")
        torch.testing.assert_close(result, torch.full((2, 2), 2.0))
        self.assertEqual(calls[0][:2], (3, "routing"))
        self.assertIs(calls[0][2], live_count)

    def test_partial_hook_install_is_cleaned_up_and_can_retry(self):
        with patch.object(
            self.decoder,
            "register_forward_pre_hook",
            side_effect=RuntimeError("hook failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "hook failed"):
                self.install()
        self.assertFalse(hasattr(self.decoder.mlp.experts, "_moe_offload_runtime"))
        self.assertNotIn("forward_hc_pre_from_prev", self.decoder.__dict__)
        self.install()
        self.owner.close()
        self.assertEqual(self.runner.capture_head_hooks, [])
        self.assertEqual(self.runner.capture_tail_hooks, [])
        self.assertFalse(hasattr(self.decoder.mlp.experts, "_moe_offload_runtime"))
        self.assertNotIn("forward_hc_pre_from_prev", self.decoder.__dict__)

    def test_capture_gates_statistics_and_serving_does_not_emit_a_second_tail(self):
        self.install()
        device = self.runtime.device_runtime
        device.allocated = lambda: 0
        device.peak = lambda: 0
        device.reset_peak = lambda: None
        with self.runtime.capture_scope(
            self.owner, self.runner, device.compute
        ) as retained:
            self.runtime._capture_head(self.runner, batch(True), 2)
            context = self.runtime.context
            self.assertFalse(bool(self.runtime.residency.update_enabled))
            self.runtime._capture_tail(self.runner, None, batch(True), 2)
            self.assertIn(context, retained)
            self.assertTrue(context._closed)
        self.assertTrue(bool(self.runtime.residency.update_enabled))
        with patch.object(
            self.runtime, "emit_step_end", wraps=self.runtime.emit_step_end
        ) as tail:
            with self.runtime.forward_scope(self.owner, batch(True)):
                pass  # Graph replay does not execute Python model hooks.
            tail.assert_not_called()
        self.assertIsNot(self.runtime._context, context)

    def test_verify_rejected_and_eager_decode_not_silently_enabled(self):
        bad = batch(True)
        bad.forward_mode = SimpleNamespace(
            is_decode_or_idle=lambda: False, is_extend_without_speculative=lambda: False
        )
        with self.assertRaisesRegex(ValueError, "ordinary extend"):
            self.runtime._phase(bad)
        with self.assertRaisesRegex(RuntimeError, "captured full graph"):
            with self.runtime.forward_scope(self.owner, batch(True)):
                self.runtime._ensure_step()

    def test_model_runner_dispatches_eager_and_graph_without_legacy_step_scope(self):
        forward = source_method(
            "python/sglang/srt/model_executor/model_runner.py",
            "ModelRunner",
            "_forward_raw",
            contextlib=contextlib,
            ModelRunnerOutput=SimpleNamespace,
            has_forward_context=lambda: True,
            get_global_dwdp_manager=lambda: None,
        )
        b = batch(True)
        b.forward_mode.is_cpu_graph = lambda: True
        b.forward_mode.is_decode = lambda: True
        b.forward_mode.is_split_prefill = lambda: False
        b.forward_mode.is_extend = lambda **kwargs: False
        b.global_num_tokens_cpu = None
        runner = SimpleNamespace(
            model=SimpleNamespace(),
            device="cpu",
            hisparse_coordinator=None,
            decode_cuda_graph_runner=None,
            _prepare_eager_forward_batch=Mock(),
            _maybe_execute_deferred_mamba_cow_and_clear=Mock(),
            eager_runner=SimpleNamespace(execute=Mock(return_value="eager")),
        )
        output = forward(runner, b, None)
        self.assertEqual((output.logits_output, output.can_run_graph), ("eager", False))
        runner.decode_cuda_graph_runner = SimpleNamespace(
            can_run_graph=lambda b: True, execute=Mock(return_value="graph")
        )
        output = forward(runner, b, None)
        self.assertEqual((output.logits_output, output.can_run_graph), ("graph", True))
        self.assertEqual(runner.eager_runner.execute.call_count, 1)
        self.assertEqual(runner._prepare_eager_forward_batch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
