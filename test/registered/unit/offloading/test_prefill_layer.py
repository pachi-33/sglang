"""Whole-layer Prefill regressions; CPU transport and native FFN test doubles."""

import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_npu_backend import Assembly
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.execution import LocalExecution
from sglang.srt.offloading.moe.layer_execution import (
    LayerExecutionRequest,
    whole_layer_bank_layout,
)
from sglang.srt.offloading.moe.layer_prefetch import LayerPrefetchPipeline
from sglang.srt.offloading.moe.layer_programs import PrefillLayerProgram
from sglang.srt.offloading.moe.npu import experts as native
from sglang.srt.offloading.moe.npu.config import NpuOffloadSettings
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.runtime import BoundMoERuntime
from sglang.srt.offloading.transfer import (
    Completion,
    ImmediateCompletion,
    TransferTicket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@dataclass
class QuantInfo:
    w13_weight: object
    w2_weight: object


class TestPrefillAssembly(unittest.TestCase):
    def test_prefill_mode_configuration(self):
        values = dict(
            cache_ratio=0,
            execution_slots=16,
            host_pool_gib=1,
            workspace_limit_bytes=1024,
            metadata_limit_bytes=1024,
            max_dispatch_rows=32,
        )
        self.assertEqual(NpuOffloadSettings(**values).prefill_mode, "layer")
        self.assertEqual(
            NpuOffloadSettings(**values, prefill_mode="wave").prefill_mode, "wave"
        )
        with self.assertRaisesRegex(ValueError, "prefill_mode"):
            NpuOffloadSettings(**values, prefill_mode="invalid")
        with self.assertRaisesRegex(ValueError, "layer_profile"):
            NpuOffloadSettings(**values, layer_profile="true")

    def test_bank_capacity_uses_only_explicitly_budgeted_pools(self):
        self.assertEqual(
            whole_layer_bank_layout(4, 8, 4), (("execution", 0, 4), ("hot", 0, 4))
        )
        self.assertEqual(
            whole_layer_bank_layout(4, 0, 8), (("execution", 0, 4), ("execution", 4, 4))
        )
        with self.assertRaisesRegex(ValueError, "execution_slots"):
            whole_layer_bank_layout(4, 8, 3)
        with self.assertRaisesRegex(ValueError, "second bank"):
            whole_layer_bank_layout(4, 3, 7)

    def test_runtime_layer_entry_keeps_pipeline_dependencies_without_global_barrier(
        self,
    ):
        p = SimpleNamespace(
            manages_layer_dependencies=True,
            layer_start=lambda layer, ctx: ctx.calls.append(("start", layer)),
        )
        runtime = SimpleNamespace(layer_programs={3: SimpleNamespace(prefill=p)})
        ctx = Context()
        ctx.phase, ctx.capturing = Phase.PREFILL, False
        BoundMoERuntime.emit_layer_start(runtime, 3, ctx)
        self.assertEqual(ctx.calls, [("start", 3)])
        p.manages_layer_dependencies = False
        BoundMoERuntime.emit_layer_start(runtime, 3, ctx)
        self.assertEqual(
            [x[0] for x in ctx.calls], ["start", "record", "wait", "start"]
        )

    def test_real_weight_bundles_and_exact_decode_cache_restore(self):
        for ratio, slots in ((0.5, 4), (0.0, 8)):
            with self.subTest(ratio=ratio):
                a = Assembly(ratio=ratio, slots=slots, local_experts=4, layer_count=4)
                self.addCleanup(a.close)
                a.backend.settings.prefill_mode = "layer"
                runtime = a.finalize()
                keys = [0, 1, 4, 5, 8, 9, 12, 13] if ratio else []
                runtime.initialize_hot_cache(keys)
                hot = a.backend.format.banks[0]
                saved = {name: tensor.clone() for name, tensor in hot.tensors.items()}
                key_to_slot = runtime.residency.key_to_slot.clone()
                generations = runtime.residency.slot_generation.clone()
                usage = runtime.cache_updates.policy_state.frequency
                usage.copy_(torch.arange(usage.numel()))
                saved_usage = usage.clone()
                runtime._decode_step_mod = 2
                owner = MoEOffloadOwner(a.config, a.catalog, runtime)
                owner.switch_phase(Phase.PREFILL)
                p = runtime.pipelines[0]
                self.assertIsInstance(p, LayerPrefetchPipeline)
                for _ in range(2):
                    ctx = runtime._new_context(Phase.PREFILL)
                    runtime._context = ctx
                    for layer in p.layer_ids:
                        p.start(layer, ctx)
                        lease = p.acquire(layer, ctx)
                        binding = p.access.wait_ready(
                            lease, ctx.compute_stream, context=ctx
                        )
                        for tensor in binding.bank.tensors.values():
                            self.assertEqual(
                                tensor[:, 0].tolist(),
                                [layer * 10 + i for i in range(4)],
                            )
                        p.retire(layer, lease, ctx.record_compute_completion(), ctx)
                        p.finish_layer(layer, ctx)
                    p.finish_forward(ctx)
                    ctx.finish().resources_idle.completion.synchronize()
                    ctx.dispose_completed()
                    runtime._context = None
                owner.switch_phase(Phase.DECODE)
                for name, tensor in hot.tensors.items():
                    torch.testing.assert_close(tensor, saved[name])
                torch.testing.assert_close(runtime.residency.key_to_slot, key_to_slot)
                torch.testing.assert_close(
                    runtime.residency.slot_generation, generations
                )
                torch.testing.assert_close(usage, saved_usage)
                self.assertEqual(runtime._decode_step_mod, 2)
                self.assertEqual(len(a.backend.format.banks), 2)
                owner.close()

    def test_whole_program_one_dispatch_and_ffn_per_layer(self):
        a = Assembly(local_experts=4, slots=4, layer_count=4)
        self.addCleanup(a.close)
        a.backend.settings.prefill_mode = "layer"
        runtime = a.finalize()
        owner = MoEOffloadOwner(a.config, a.catalog, runtime)
        self.addCleanup(owner.close)
        pipeline = runtime.pipelines[0]
        calls = []
        raw = SimpleNamespace(
            format="deepep_normal",
            hidden_states=torch.ones(6, 4, dtype=torch.bfloat16),
            hidden_states_scale=None,
            num_recv_tokens_per_expert=[1, 2, 0, 3],
        )

        def dispatch(x, routing, context):
            calls.append("dispatch")
            return native.NPUDispatch(
                raw, TransferTicket(ImmediateCompletion()), 4, object()
            )

        def combine(layout, context):
            calls.append("combine")
            return layout.output

        dispatcher = SimpleNamespace(
            dispatch=dispatch, combine=combine, finish=lambda output, context: output
        )
        layout = native.NPUDispatchLayout(4, 4, 8, "cpu", runtime.ops.error_status)

        def ffn(layer, inputs, binding, context):
            calls.append(("ffn", layer))
            self.assertEqual(inputs.expert_tokens.tolist(), [1, 2, 0, 3])
            for tensor in binding.bank.tensors.values():
                self.assertEqual(
                    tensor[:, 0].tolist(), [layer * 10 + i for i in range(4)]
                )
            return LocalExecution(
                inputs.hidden_states * layer, context.record_compute_completion()
            )

        for layer, programs in runtime.layer_programs.items():
            programs.prefill = PrefillLayerProgram(
                dispatcher,
                layout,
                ffn,
                pipeline.access,
                compute_scope=lambda ctx: nullcontext(),
                pipeline=pipeline,
            )
        mode = SimpleNamespace(
            is_extend_without_speculative=lambda: True, is_decode_or_idle=lambda: False
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            input_ids=torch.ones(6),
            batch_size=1,
            moe_num_token_non_padded=lambda: None,
        )
        with patch.object(native, "_compute_scope", lambda *args: nullcontext()):
            with runtime.forward_scope(owner, batch):
                for layer in pipeline.layer_ids:
                    runtime.emit_layer_start(layer, runtime.context)
                    result = runtime.run_layer(layer, raw.hidden_states, None)
                    torch.testing.assert_close(result, raw.hidden_states * layer)
        self.assertEqual(
            calls,
            [
                call
                for layer in pipeline.layer_ids
                for call in ("dispatch", ("ffn", layer), "combine")
            ],
        )


class Fence(Completion):
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def wait_on(self, stream):
        self.calls.append(("wait", self.name, stream))

    def synchronize(self):
        raise AssertionError("No host synchronization in layer planning")

    def done(self):
        raise AssertionError("No completion polling in layer planning")


class Context:
    phase = Phase.DECODE
    capturing = True
    compute_stream = "compute"
    copy_stream = "copy"

    def __init__(self):
        self.invocation_id = object()
        self.calls, self.retained = [], []

    def retain(self, value):
        self.retained.append(value)

    retain_completion = retain

    def record_compute_completion(self):
        name = "compute-" + str(len(self.calls))
        self.calls.append(("record", name))
        return TransferTicket(Fence(self.calls, name))

    def copied(self):
        return TransferTicket(Fence(self.calls, "copy-done"))

    def wait_for(self, ticket, stream):
        ticket.completion.wait_on(stream)

    def temporary_scope(self):
        return nullcontext()


class TensorTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        self.calls.append((func, args))
        return result


class TestLayerPrefetch(unittest.TestCase):
    def make_pipeline(self, layers=(3, 5, 8, 9)):
        ctx = Context()
        ctx.phase, ctx.capturing = Phase.PREFILL, False
        access = SimpleNamespace()

        def stage(request, context):
            context.calls.append(("load", request.layer_id, request.bank_id))
            return SimpleNamespace(
                layer=request.layer_id,
                read_done=None,
                ticket=TransferTicket(
                    Fence(context.calls, f"loaded-{request.layer_id}")
                ),
            )

        def release(lease, done, context):
            if lease.read_done is not None:
                raise RuntimeError("Duplicate retirement")
            lease.read_done = done

        access.stage, access.release_after = stage, release
        p = LayerPrefetchPipeline(
            access,
            layers,
            make_request=lambda layer, bank, _: LayerExecutionRequest(
                layer, None, Phase.PREFILL, bank
            ),
            record_copy_completion=lambda c: TransferTicket(Fence(c.calls, "abort")),
        )
        return p, ctx

    def test_lookahead_and_per_bank_dependencies_without_poll_or_host_wait(self):
        p, ctx = self.make_pipeline()
        p.start(3, ctx)
        self.assertEqual(
            [x for x in ctx.calls if x[0] == "load"], [("load", 3, 0), ("load", 5, 1)]
        )
        for layer in p.layer_ids:
            if layer != 3:
                p.start(layer, ctx)
            lease = p.acquire(layer, ctx)
            done = TransferTicket(Fence(ctx.calls, f"reader-{layer}"))
            p.retire(layer, lease, done, ctx)
            p.finish_layer(layer, ctx)
        p.finish_forward(ctx)
        calls = ctx.calls
        self.assertLess(
            calls.index(("wait", "reader-3", "copy")), calls.index(("load", 8, 0))
        )
        self.assertLess(
            calls.index(("wait", "reader-5", "copy")), calls.index(("load", 9, 1))
        )
        self.assertFalse(any(x[0] == "wait" and x[-1] == "compute" for x in calls))
        # A new forward starts a fresh load sequence; no stale ready events.
        next_ctx = Context()
        next_ctx.phase, next_ctx.capturing = Phase.PREFILL, False
        p.start(3, next_ctx)
        self.assertEqual(len([x for x in next_ctx.calls if x[0] == "load"]), 2)
        p.abort(next_ctx)
        p.close()

    def test_wrong_order_incomplete_forward_and_partial_failure(self):
        p, ctx = self.make_pipeline()
        with self.assertRaises(RuntimeError):
            p.start(5, ctx)
        p.start(3, ctx)
        with self.assertRaises(RuntimeError):
            p.start(3, ctx)
        with self.assertRaises(RuntimeError):
            p.finish_forward(ctx)
        p.abort(ctx)
        with self.assertRaises(RuntimeError):
            p.start(3, ctx)
        p.close()
        p, ctx = self.make_pipeline()
        original = p.access.stage

        def fail(request, context):
            if request.layer_id == 5:
                raise RuntimeError("partial copy failure")
            return original(request, context)

        p.access.stage = fail
        with self.assertRaisesRegex(RuntimeError, "partial copy"):
            p.start(3, ctx)
        self.assertEqual(p.abort(ctx).completion.name, "abort")
        p.close()


class TestLayerWorkspace(unittest.TestCase):
    def test_compact_prefill_counts_and_output_without_reductions_or_roundtrip(self):
        for dtype in (torch.bfloat16, torch.float16):
            with (
                self.subTest(dtype=dtype),
                patch.object(native, "_compute_scope", lambda *args: nullcontext()),
            ):
                error = torch.zeros((), dtype=torch.int32)
                adapter = native.NPUDispatchLayout(4, 4, 8, "cpu", error)
                adapter._owned["counts"].fill_(-99)
                adapter._owned["offsets"].fill_(-99)
                adapter._owned["output"].fill_(-99)
                ctx = Context()
                ctx.phase, ctx.capturing = Phase.PREFILL, False
                x = torch.randn(8, 4).to(dtype)
                scale = torch.arange(8, dtype=torch.float32)
                raw = SimpleNamespace(
                    format="deepep_normal",
                    hidden_states=x,
                    hidden_states_scale=scale,
                    num_recv_tokens_per_expert=[2, 0, 5, 1],
                )
                with TensorTrace() as trace:
                    layout = adapter.bind(
                        native.NPUDispatch(
                            raw, TransferTicket(ImmediateCompletion()), 4, object()
                        ),
                        ctx,
                        whole_layer=True,
                    )
                    inputs = adapter.select_layer(layout, ctx)
                reductions = {"aten.any.default", "aten.sum.default", "aten.cumsum.out"}
                self.assertFalse(reductions & {str(call[0]) for call in trace.calls})
                self.assertEqual(inputs.row_count, 8)
                self.assertEqual(inputs.hidden_states.data_ptr(), x.data_ptr())
                self.assertEqual(
                    inputs.hidden_states_scale.data_ptr(), scale.data_ptr()
                )
                torch.testing.assert_close(
                    inputs.expert_tokens, torch.tensor([2, 0, 5, 1])
                )
                self.assertTrue((adapter._owned["counts"] == -99).all())
                self.assertTrue((adapter._owned["offsets"] == -99).all())

                native_output = x * 2

                def run(data, quant, state):
                    self.assertIs(data.expert_tokens, inputs.expert_tokens)
                    return SimpleNamespace(hidden_states=native_output)

                runner = native.NPUExpertRunner(
                    None, _core=SimpleNamespace(run=run), _input_type=SimpleNamespace
                )
                bank = native.NPUExpertBank(
                    QuantInfo(torch.ones(4, 1, 1), torch.ones(4, 1, 1)),
                    torch.ones(4, dtype=torch.bool),
                    error,
                )
                with TensorTrace() as trace:
                    result = runner(3, inputs, SimpleNamespace(expert_bank=bank), ctx)
                    adapter.write_layer(layout, result.output, ctx)
                    output = adapter.combine_input(layout, ctx).output
                self.assertFalse(
                    (
                        reductions
                        | {
                            "aten.where.self",
                            "aten.arange.default",
                            "aten.copy_.default",
                        }
                    )
                    & {str(call[0]) for call in trace.calls}
                )
                self.assertIs(output, native_output)
                torch.testing.assert_close(output, x * 2)
                # A compact result never clears or writes the shared output pool.
                self.assertTrue((adapter._owned["output"] == -99).all())

    def test_prefill_fault_latch_suppresses_unwritten_ffn_output(self):
        for used, late_fault in (
            (0, False),
            (3, False),
            (8, False),
            (3, True),
            (8, True),
        ):
            with (
                self.subTest(used=used, late_fault=late_fault),
                patch.object(native, "_compute_scope", lambda *args: nullcontext()),
            ):
                error = torch.tensor(0 if late_fault else 256, dtype=torch.int32)
                adapter = native.NPUDispatchLayout(4, 4, 8, "cpu", error)
                adapter._owned["output"].fill_(float("nan"))
                ctx = Context()
                ctx.phase, ctx.capturing = Phase.PREFILL, False
                raw = SimpleNamespace(
                    format="deepep_normal",
                    hidden_states=torch.ones(8, 4, dtype=torch.bfloat16),
                    hidden_states_scale=None,
                    num_recv_tokens_per_expert=[0, used, 0, 0],
                )
                layout = adapter.bind(
                    native.NPUDispatch(
                        raw, TransferTicket(ImmediateCompletion()), 4, object()
                    ),
                    ctx,
                    whole_layer=True,
                )
                inputs = adapter.select_layer(layout, ctx)
                if late_fault:
                    # The runner must observe faults reported after dispatch,
                    # when the program joins weight preparation before FFN.
                    error.fill_(256)
                calls = []

                def run(data, quant, state):
                    calls.append("ffn")
                    self.assertEqual(data.expert_tokens.tolist(), [0] * 4)
                    return SimpleNamespace(
                        hidden_states=torch.full_like(data.hidden_states, float("nan"))
                    )

                runner = native.NPUExpertRunner(
                    None, _core=SimpleNamespace(run=run), _input_type=SimpleNamespace
                )
                bank = native.NPUExpertBank(
                    QuantInfo(torch.ones(4, 1, 1), torch.ones(4, 1, 1)),
                    torch.ones(4, dtype=torch.bool),
                    error,
                )
                result = runner(3, inputs, SimpleNamespace(expert_bank=bank), ctx)
                adapter.write_layer(layout, result.output, ctx)
                output = adapter.combine_input(layout, ctx).output
                torch.testing.assert_close(
                    output, torch.zeros(8, 4, dtype=output.dtype)
                )
                self.assertEqual(int(error), 256)
                self.assertEqual(calls, ["ffn"] if used else [])

    def test_prefill_reuses_output_pool_without_stale_padding(self):
        with patch.object(native, "_compute_scope", lambda *args: nullcontext()):
            adapter = native.NPUDispatchLayout(
                4, 4, 8, "cpu", torch.zeros((), dtype=torch.int32)
            )
            ctx = Context()
            ctx.phase, ctx.capturing = Phase.PREFILL, False
            for used in (8, 3, 0, 6, 8):
                with self.subTest(used=used):
                    raw = SimpleNamespace(
                        format="deepep_normal",
                        hidden_states=torch.ones(8, 4, dtype=torch.bfloat16),
                        hidden_states_scale=None,
                        num_recv_tokens_per_expert=[used, 0, 0, 0],
                    )
                    layout = adapter.bind(
                        native.NPUDispatch(
                            raw, TransferTicket(ImmediateCompletion()), 4, object()
                        ),
                        ctx,
                        whole_layer=True,
                    )
                    inputs = adapter.select_layer(layout, ctx)
                    result = inputs.hidden_states * (used + 1)
                    adapter.write_layer(layout, result, ctx)
                    output = adapter.combine_input(layout, ctx).output
                    expected = torch.zeros_like(output)
                    expected[:used] = used + 1
                    torch.testing.assert_close(output, expected)
                    if used == 8:
                        self.assertIs(output, result)
                    else:
                        self.assertEqual(
                            output.data_ptr(), adapter._owned["output"].data_ptr()
                        )
