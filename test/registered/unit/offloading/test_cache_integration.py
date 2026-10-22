"""Production D/F/C + cache + prefill pipeline; only hardware operations are doubled."""

import json
import sys
import tempfile
import unittest
from collections import namedtuple
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from test_cache_model_binding import CpuCore
from test_npu_backend import Assembly

from sglang.srt.offloading.moe.config import Phase
from sglang.srt.offloading.moe.npu import experts
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.periodic_lfu import PeriodicLFUConfig, PeriodicLFUPolicy
from sglang.srt.offloading.transfer import Completion, TransferTicket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

Routing = namedtuple("Routing", "topk_ids topk_weights")


class CpuDeepEP:
    """Test collective: rank 1 receives its E4..E7 assignments in expert order."""

    def __init__(self):
        self.calls = []
        self.handle = None

    def dispatch(self, hidden_states, topk_output):
        self.calls.append("D")
        ids, weights = topk_output
        tokens, factors, counts = [], [], []
        for expert in range(4, 8):
            rows, lanes = torch.where(ids == expert)
            tokens.extend(rows.tolist())
            factors.extend(weights[rows, lanes].tolist())
            counts.append(rows.numel())
        self.handle = (tokens, factors, hidden_states.shape)
        x = torch.zeros(ids.numel(), hidden_states.shape[-1], dtype=hidden_states.dtype)
        if tokens:
            x[: len(tokens)] = hidden_states[tokens]
        return SimpleNamespace(
            hidden_states=x,
            hidden_states_scale=None,
            masked_m=torch.tensor(counts),
            format="deepep_ll",
            topk_ids=ids,
            topk_weights=weights,
        )

    def combine(self, combine_input):
        self.calls.append("C")
        tokens, factors, shape = self.handle
        out = torch.zeros(shape, dtype=combine_input.hidden_states.dtype)
        for row, (token, weight) in enumerate(zip(tokens, factors)):
            out[token] += combine_input.hidden_states[row] * weight
        self.handle = None
        return out


class Deferred(Completion):
    def __init__(self, action):
        self.action, self.ready = action, False

    def wait_on(self, stream):
        self.synchronize()

    def synchronize(self):
        if not self.ready:
            self.action()
            self.ready = True

    def done(self):
        return self.ready


class TestNativeCacheIntegration(unittest.TestCase):
    def setUp(self):
        stack = self.stack = ExitStack()
        self.addCleanup(stack.close)
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        profile = Path(directory) / "profile.json"
        profile.write_text(
            json.dumps({"model_id": "toy", "layers": {"3": {"4": 100, "6": 90}}})
        )
        self.assembly = Assembly(slots=2, group=1, layer_count=1, local_experts=4)
        stack.callback(self.assembly.close)
        self.assembly.backend.settings.offline_model_id = "toy"
        self.assembly.policy = PeriodicLFUPolicy(
            PeriodicLFUConfig(str(profile)), self.assembly.catalog
        )
        module = ModuleType("sglang.srt.layers.moe.token_dispatcher.deepep")
        module.DeepEPDispatcher = CpuDeepEP
        module.DeepEPLLCombineInput = lambda x, ids, weights: SimpleNamespace(
            hidden_states=x
        )
        module.DeepEPNormalCombineInput = module.DeepEPLLCombineInput
        previous = sys.modules.get(module.__name__)
        sys.modules[module.__name__] = module

        def restore_module():
            if previous is None:
                sys.modules.pop(module.__name__, None)
            else:
                sys.modules[module.__name__] = previous

        stack.callback(restore_module)
        stack.enter_context(
            patch.dict(
                "os.environ",
                {
                    "MOE_ENABLE_TOPK_NEG_ONE": "1",
                    "MOE_EXPERT_TOKEN_NUMS_TYPE": "1",
                    "DEEP_USE_MODE": "default",
                },
            )
        )
        stack.enter_context(
            patch.object(experts, "_compute_scope", lambda *args: nullcontext())
        )
        runner_type = experts.NPUExpertRunner
        stack.enter_context(
            patch.object(
                experts,
                "NPUExpertRunner",
                side_effect=lambda *args: runner_type(
                    None, _core=CpuCore(), _input_type=SimpleNamespace
                ),
            )
        )
        self.native = CpuDeepEP()
        layer = self.assembly.layers[0][0].mlp.experts
        layer.dispatcher = self.native
        layer.reduce_results = False
        layer.moe_ep_size, layer.moe_tp_size = 2, 1
        layer.num_local_experts = 4
        layer.moe_runner_config = SimpleNamespace(hidden_size=2)
        # Assembly still doubles byte transfers. Build the real layer programs.
        build = experts.build_npu_layer_programs
        self.assembly.build = lambda layer, **kwargs: build(layer, **kwargs)
        self.runtime = self.assembly.finalize()
        self.owner = MoEOffloadOwner(
            self.assembly.config, self.assembly.catalog, self.runtime
        )
        stack.callback(self.owner.close)
        self.x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16)
        self.routing = Routing(
            torch.tensor([[5, 5], [6, 7], [4, 4]]),
            torch.tensor([[0.25, 0.75], [0.5, 0.5], [1.0, 1.0]]),
        )
        self.expected = self.x * torch.tensor([31, 32.5, 0])[:, None]

    def decode(self):
        # Execute the tensor operations that a real graph would replay.
        ctx = self.runtime._new_context(Phase.DECODE, capturing=True)
        ctx.num_token_non_padded = torch.tensor(2, dtype=torch.int32)
        self.runtime._context = ctx
        try:
            self.runtime.emit_step_begin(ctx)
            out = self.runtime.decode_program.run(3, self.x, self.routing, ctx)
            tail = self.runtime.emit_step_end(ctx)
            ctx.finish(residency_ready=tail)
            return out
        finally:
            self.runtime._context = None

    def test_dispatch_once_masked_batch_and_periodic_replacement(self):
        pointers = [
            t.data_ptr()
            for bank in self.assembly.backend.format.banks
            for t in bank.tensors.values()
        ]
        for _ in range(4):
            torch.testing.assert_close(self.decode(), self.expected.to(torch.bfloat16))
        self.assertEqual(self.native.calls, ["D", "C"] * 4)
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [1, 2])
        torch.testing.assert_close(
            self.runtime.cache_updates.policy_state.usage,
            torch.tensor([0, 8, 4, 4]) * 0.7,
        )
        torch.testing.assert_close(self.decode(), self.expected.to(torch.bfloat16))
        self.assertEqual(int(self.runtime.cache_stats.hits), 3)
        self.assertEqual(int(self.runtime.cache_stats.misses), 1)
        self.assertEqual(
            pointers,
            [
                t.data_ptr()
                for bank in self.assembly.backend.format.banks
                for t in bank.tensors.values()
            ],
        )

    def test_decode_prefill_decode_restores_slots_and_cycle(self):
        torch.testing.assert_close(self.decode(), self.expected.to(torch.bfloat16))
        state = self.runtime.cache_updates.policy_state
        usage, keys = state.usage.clone(), self.runtime.residency.slot_to_key.clone()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode_or_idle=lambda: False,
                is_extend_without_speculative=lambda: True,
            ),
            input_ids=torch.ones(3),
            moe_num_token_non_padded=lambda: torch.tensor(2),
        )
        with self.runtime.forward_scope(self.owner, batch):
            self.runtime.emit_layer_start(3, self.runtime.context)
            out = self.runtime.run_layer(3, self.x, self.routing)
        torch.testing.assert_close(out, self.expected.to(torch.bfloat16))
        torch.testing.assert_close(state.usage, usage)
        self.owner.switch_phase(Phase.DECODE)
        torch.testing.assert_close(self.runtime.residency.slot_to_key, keys)
        for tensor in self.assembly.backend.format.banks[0].tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [30, 32])
        self.assertEqual(int(state.step_mod), 1)
        torch.testing.assert_close(self.decode(), self.expected.to(torch.bfloat16))
        self.assertEqual(int(state.step_mod), 2)
        self.assertEqual(self.native.calls, ["D", "C"] * 3)

    def test_auxiliary_reader_finishes_before_reusing_execution_bank(self):
        calls = []
        program = self.runtime.layer_programs[3].decode
        original = program.local_program.run_experts

        def run(*args):
            result = original(*args)
            result_type = type(result)
            return result_type(
                result.output, TransferTicket(Deferred(lambda: calls.append("read")))
            )

        program.local_program.run_experts = run
        transfer = self.runtime.cache_updates.access.transfer_backend
        copy = transfer.materialize_into
        with patch.object(
            transfer,
            "materialize_into",
            side_effect=lambda *args: (calls.append("copy"), copy(*args))[1],
        ):
            self.decode()
        # Two execution waves, followed by the periodic-update graph node.
        self.assertEqual(calls[:4], ["copy", "read", "copy", "read"])

    def test_unneeded_wave_lanes_do_not_copy_or_count_padding(self):
        self.routing = Routing(
            torch.tensor([[4, 4], [-1, -1], [7, 7]]), torch.ones(3, 2)
        )
        out = self.decode()
        torch.testing.assert_close(
            out, self.x * torch.tensor([60, 0, 0], dtype=torch.bfloat16)[:, None]
        )
        self.assertEqual(
            self.runtime.residency.step_access_count.tolist(), [2, 0, 0, 0]
        )
        self.assertEqual(int(self.runtime.cache_stats.accesses), 2)
        self.assertEqual(int(self.runtime.residency.execution_valid.sum()), 0)

    def test_hot_mapping_is_published_only_after_complete_bundle_copy(self):
        ctx = self.runtime._new_context(Phase.DECODE, capturing=True)
        self.runtime._context = ctx
        self.addCleanup(setattr, self.runtime, "_context", None)
        access = self.runtime.cache_updates.access
        transfer = access.transfer_backend
        copy = transfer.materialize_into
        observed = []

        def deferred_copy(plan, binding, stream):
            def complete():
                self.assertEqual(int(self.runtime.residency.key_to_slot[1]), -1)
                self.assertFalse(bool(self.runtime.residency.slot_valid[0]))
                copy(plan, binding, stream).completion.synchronize()
                observed.append("copied")

            return TransferTicket(Deferred(complete))

        with patch.object(transfer, "materialize_into", side_effect=deferred_copy):
            access.apply_update(torch.tensor([1, 2]), ctx)
        self.assertEqual(observed, ["copied"])
        self.assertEqual(self.runtime.residency.slot_to_key.tolist(), [1, 2])
        for tensor in self.assembly.backend.format.banks[0].tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [31, 32])

    def test_failed_hot_copy_never_publishes_new_key(self):
        ctx = self.runtime._new_context(Phase.DECODE, capturing=True)
        self.runtime._context = ctx
        self.addCleanup(setattr, self.runtime, "_context", None)
        access = self.runtime.cache_updates.access
        with patch.object(
            access.transfer_backend,
            "materialize_into",
            side_effect=RuntimeError("copy failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                access.apply_update(torch.tensor([1, 2]), ctx)
        self.assertEqual(int(self.runtime.residency.key_to_slot[1]), -1)
        self.assertFalse(bool(self.runtime.residency.slot_valid[0]))
        self.assertEqual(int(self.runtime.residency.key_to_slot[2]), 1)


if __name__ == "__main__":
    unittest.main()
