"""Framework entry points against the single native runtime, with CPU device doubles."""

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from test_npu_backend import Assembly

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy
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
            self.runtime._phase(bad, has_global_work=True, current_phase=Phase.DECODE)
        with self.assertRaisesRegex(RuntimeError, "captured full graph"):
            with self.runtime.forward_scope(self.owner, batch(True)):
                self.runtime._ensure_step()

    def test_global_prefill_flag_does_not_enable_speculative_modes(self):
        for extending in (False, True):
            with self.subTest(is_extend_in_batch=extending):
                ordinary = batch(True)
                ordinary.is_extend_in_batch = extending
                self.assertIs(
                    self.runtime._phase(
                        ordinary, has_global_work=True, current_phase=Phase.DECODE
                    ),
                    Phase.PREFILL if extending else Phase.DECODE,
                )
                unsupported = batch(True)
                unsupported.is_extend_in_batch = extending
                unsupported.forward_mode = SimpleNamespace(
                    is_decode_or_idle=lambda: False,
                    is_extend_without_speculative=lambda: False,
                )
                with self.assertRaisesRegex(ValueError, "ordinary extend"):
                    self.runtime._phase(
                        unsupported, has_global_work=True, current_phase=Phase.DECODE
                    )

    def test_global_prefill_eager_path_preserves_cache_and_lfu_on_local_decode_or_idle(
        self,
    ):
        assembly = Assembly(layer_count=1, local_experts=4)
        assembly.policy = PeriodicLFUPolicy(PeriodicLFUConfig(), assembly.catalog)
        runtime = assembly.finalize()
        owner = MoEOffloadOwner(assembly.config, assembly.catalog, runtime)
        self.addCleanup(owner.close)
        policy = runtime.cache_updates.policy_state
        policy.usage.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        policy.step_mod.fill_(3)  # An accidental decode tail would update LFU.
        saved_policy = {name: value.clone() for name, value in vars(policy).items()}
        saved_keys = runtime.residency.slot_to_key.clone()
        hot = assembly.backend.format.banks[0].tensors
        saved_weights = {name: value.clone() for name, value in hot.items()}

        def prefill(layer, hidden, routing, context):
            self.assertIs(context.phase, Phase.PREFILL)
            self.assertIsNotNone(runtime.snapshot._snapshot)
            # Simulate prefill borrowing the hot bank. The subsequent decode
            # transition must reload these bytes before restoring its mapping.
            for tensor in hot.values():
                tensor.fill_(199)
            runtime.residency.step_access_count.fill_(10)
            return hidden

        runtime.layer_programs[3].prefill = SimpleNamespace(run=prefill)
        forward = source_method(
            "python/sglang/srt/model_executor/model_runner.py",
            "ModelRunner",
            "_forward_raw",
            contextlib=contextlib,
            ModelRunnerOutput=SimpleNamespace,
            has_forward_context=lambda: True,
            get_global_dwdp_manager=lambda: None,
        )
        runner = SimpleNamespace(
            model=SimpleNamespace(),
            device="cpu",
            hisparse_coordinator=None,
            decode_cuda_graph_runner=SimpleNamespace(
                can_run_graph=lambda b: b.can_run_decode_cuda_graph,
                execute=Mock(side_effect=AssertionError("Global prefill used graph")),
            ),
            _prepare_eager_forward_batch=Mock(),
            _maybe_execute_deferred_mamba_cow_and_clear=Mock(),
            eager_runner=SimpleNamespace(
                execute=lambda b, **kwargs: runtime.run_layer(3, b.input_ids, None)
            ),
        )
        for idle in (False, True):
            with self.subTest(local_mode="idle" if idle else "decode"):
                b = batch(True)
                b.is_extend_in_batch = True
                b.can_run_decode_cuda_graph = False  # Scheduler's global vote.
                b.original_global_num_tokens_cpu = [0 if idle else 2, 8]
                b.input_ids = torch.ones(0 if idle else 2, dtype=torch.int64)
                b.forward_mode.is_cpu_graph = lambda: True
                b.forward_mode.is_decode = lambda: not idle
                b.forward_mode.is_split_prefill = lambda: False
                b.forward_mode.is_extend = lambda **kwargs: False
                b.global_num_tokens_cpu = None  # Skip unrelated logits postprocess.
                with runtime.forward_scope(owner, b):
                    output = forward(runner, b, None)
                self.assertFalse(output.can_run_graph)
                self.assertIs(owner.phase, Phase.PREFILL)
                for name, expected in saved_policy.items():
                    torch.testing.assert_close(getattr(policy, name), expected)

                # Graph replay does not execute Python layer/tail hooks. This
                # serving scope still performs the real prefill->decode restore.
                with runtime.forward_scope(owner, batch(True)):
                    pass
                self.assertIs(owner.phase, Phase.DECODE)
                self.assertIsNone(runtime.snapshot._snapshot)
                torch.testing.assert_close(runtime.residency.slot_to_key, saved_keys)
                for name, expected in saved_weights.items():
                    torch.testing.assert_close(hot[name], expected)
                for name, expected in saved_policy.items():
                    torch.testing.assert_close(getattr(policy, name), expected)
        self.assertEqual(runner._prepare_eager_forward_batch.call_count, 2)
        runner.decode_cuda_graph_runner.execute.assert_not_called()

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
