import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.moe.expert_offload import (
    CachedExpertLayerView,
    ExpertOffloadContext,
    ExpertOffloadSpec,
    ExpertOffloadTensorSpec,
    OffloadedFusedMoEMethod,
)
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer_module
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.awq.awq import AWQMoEMethod
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_loader import loader as model_loader_module
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-b-test-cpu-intel")


class _ToyMethod(FusedMoEMethodBase):
    def __init__(self, *, expert_axis=0, supports_slot_remap=True):
        self.loader = None
        self.post_load_calls = 0
        self.expert_axis = expert_axis
        self.supports_slot_remap = supports_slot_remap

    def get_expert_offload_spec(self):
        return ExpertOffloadSpec(
            tensors=(
                ExpertOffloadTensorSpec("w13_weight", ("w1", "w3"), self.expert_axis),
                ExpertOffloadTensorSpec("w2_weight", ("w2",), self.expert_axis),
            ),
            supports_slot_remap=self.supports_slot_remap,
        )

    def create_weights(self, layer, num_experts, **kwargs):
        self.loader = kwargs["weight_loader"]
        layer.register_parameter(
            "w13_weight", nn.Parameter(torch.empty(num_experts, 2, 2))
        )
        layer.register_parameter(
            "w2_weight", nn.Parameter(torch.empty(num_experts, 2, 2))
        )

    def create_moe_runner(self, layer, moe_runner_config):
        return

    def apply(self, layer, dispatch_output):
        raise AssertionError("load-only test method must not execute")

    def process_weights_after_loading(self, layer):
        self.post_load_calls += 1


class _UnsupportedMethod(FusedMoEMethodBase):
    def create_moe_runner(self, layer, moe_runner_config):
        return

    def apply(self, layer, dispatch_output):
        raise AssertionError("unsupported test method must not execute")


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_id = 0
        self.top_k = 1
        self.num_local_experts = 2

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return expert_id

    def named_per_expert_tensors(self, num_experts):
        return [("side_scale", self.side_scale)]


class _ToyModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def load_weights(self, weights):
        return


class TestExpertOffload(unittest.TestCase):
    def _create_method(self, *, pin_host_experts=False):
        layer = _ToyLayer()
        layer.register_buffer("side_scale", torch.ones(2, 1))
        inner = _ToyMethod()
        context = ExpertOffloadContext(pin_host_experts=pin_host_experts)
        method = OffloadedFusedMoEMethod(inner, context)

        def original_loader(param, loaded, weight_name, shard_id, expert_id):
            param.data[expert_id].copy_(loaded)
            return "consumed"

        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=2,
            intermediate_size_per_partition=2,
            params_dtype=torch.float32,
            weight_loader=original_loader,
        )
        return layer, inner, method

    def test_coverage_loader_preserves_original_return_value(self):
        layer, inner, _ = self._create_method()
        assert inner.loader is not None
        self.assertEqual(
            inner.loader(
                layer.w13_weight,
                torch.ones(2, 2),
                "w13_weight",
                "w1",
                0,
            ),
            "consumed",
        )

    def _load_complete_weights(self, layer, inner):
        assert inner.loader is not None
        for expert_id in range(2):
            for shard_id in ("w1", "w3"):
                inner.loader(
                    layer.w13_weight,
                    torch.full((2, 2), expert_id + 1.0),
                    "w13_weight",
                    shard_id,
                    expert_id,
                )
            inner.loader(
                layer.w2_weight,
                torch.full((2, 2), expert_id + 3.0),
                "w2_weight",
                "w2",
                expert_id,
            )

    def test_keeps_original_loader_and_captures_cpu_post_load_sources(self):
        layer, inner, method = self._create_method()
        self.assertEqual(layer.w13_weight.device.type, "cpu")
        self._load_complete_weights(layer, inner)

        method.finalize_weight_loading(layer)
        method.process_weights_after_loading(layer)
        method.finalize_post_load(layer)

        self.assertEqual(inner.post_load_calls, 1)
        self.assertEqual(
            set(method.source_tensors), {"side_scale", "w13_weight", "w2_weight"}
        )
        self.assertIs(
            method.context.backend.host_layers[0].tensors["w13_weight"],
            layer.w13_weight,
        )
        self.assertTrue(torch.equal(layer.w2_weight[1], torch.full((2, 2), 4.0)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_pin_host_experts_replaces_storage_but_preserves_parameters(self):
        layer, inner, method = self._create_method(pin_host_experts=True)
        self._load_complete_weights(layer, inner)
        original_w13 = layer.w13_weight
        original_w2 = layer.w2_weight

        method.finalize_weight_loading(layer)
        method.process_weights_after_loading(layer)
        method.finalize_post_load(layer)

        self.assertIs(layer.w13_weight, original_w13)
        self.assertIs(layer.w2_weight, original_w2)
        self.assertTrue(layer.w13_weight.is_pinned())
        self.assertTrue(layer.w2_weight.is_pinned())
        self.assertTrue(layer.side_scale.is_pinned())
        host = method.context.backend.host_layers[0]
        self.assertEqual(host.pinned_bytes, host.total_bytes)

    def test_rejects_incomplete_shard_coverage(self):
        layer, inner, method = self._create_method()
        assert inner.loader is not None
        inner.loader(layer.w13_weight, torch.ones(2, 2), "w13_weight", "w1", 0)

        with self.assertRaisesRegex(RuntimeError, "coverage is incomplete"):
            method.finalize_weight_loading(layer)

    def test_cached_layer_view_overrides_only_expert_attributes(self):
        layer = _ToyLayer()
        layer.register_parameter("w13_weight", nn.Parameter(torch.zeros(2, 1)))
        layer.other = "unchanged"
        cached = torch.ones(3, 1)

        view = CachedExpertLayerView(layer, {"w13_weight": cached})

        self.assertIs(view.w13_weight, cached)
        self.assertEqual(view.other, "unchanged")

    def test_context_rejects_unsupported_method_and_forwards_runner(self):
        context = ExpertOffloadContext()
        with self.assertRaisesRegex(RuntimeError, "does not declare"):
            context.wrap(_UnsupportedMethod())

        inner = _ToyMethod()
        inner.runner = object()
        method = OffloadedFusedMoEMethod(inner, context)
        self.assertIs(method.runner, inner.runner)

    def test_wrapper_replaces_module_quant_method_without_registering_inner(self):
        """FusedMoE may initially register a module-backed quant method."""
        owner = nn.Module()
        inner = UnquantizedFusedMoEMethod()
        self.assertIsInstance(inner, nn.Module)
        owner.quant_method = inner

        wrapped = OffloadedFusedMoEMethod(inner, ExpertOffloadContext())
        owner.quant_method = wrapped

        self.assertIs(owner.quant_method, wrapped)
        self.assertIs(wrapped.inner, inner)
        self.assertTrue(wrapped.use_triton_kernels == inner.use_triton_kernels)
        # The delegate must not become ``quant_method.inner`` in the model
        # traversal; otherwise generic post-load hooks could visit it twice.
        self.assertNotIn("quant_method.inner", dict(owner.named_modules()))
        self.assertEqual(list(owner.children()), [wrapped])

    def test_unquantized_capability_and_default_none_contract(self):
        spec = UnquantizedFusedMoEMethod().get_expert_offload_spec()
        self.assertTrue(spec.supports_slot_remap)
        self.assertEqual(
            [(item.name, item.shard_ids, item.expert_axis) for item in spec.tensors],
            [("w13_weight", ("w1", "w3"), 0), ("w2_weight", ("w2",), 0)],
        )
        self.assertIsNone(_UnsupportedMethod().get_expert_offload_spec())

    def test_awq_capability_contract(self):
        spec = AWQMoEMethod(SimpleNamespace(weight_bits=4)).get_expert_offload_spec()
        self.assertTrue(spec.supports_slot_remap)
        self.assertEqual(
            [(item.name, item.shard_ids) for item in spec.tensors],
            [
                ("w13_qweight", ("w1", "w3")),
                ("w13_scales", ("w1", "w3")),
                ("w13_qzeros", ("w1", "w3")),
                ("w2_qweight", ("w2",)),
                ("w2_scales", ("w2",)),
                ("w2_qzeros", ("w2",)),
            ],
        )

    def test_wrapper_requires_axis_zero_and_slot_remapping(self):
        context = ExpertOffloadContext()
        with self.assertRaisesRegex(ValueError, "physical expert-slot"):
            OffloadedFusedMoEMethod(_ToyMethod(supports_slot_remap=False), context)
        with self.assertRaisesRegex(ValueError, "expert_axis=0"):
            OffloadedFusedMoEMethod(_ToyMethod(expert_axis=1), context)

    def test_device_loading_context_keeps_auto_pin_and_allows_override(self):
        seen_pin_memory = []

        @contextmanager
        def fake_stage(module, target_device, *, pin_memory):
            seen_pin_memory.append(pin_memory)
            yield module

        module = nn.Module()
        with (
            patch.object(model_loader_module, "stage_module_for_post_load", fake_stage),
            patch.object(
                model_loader_module, "is_pin_memory_available", return_value=True
            ),
        ):
            with model_loader_module.device_loading_context(
                module, torch.device("cuda")
            ):
                pass
            with model_loader_module.device_loading_context(
                module, torch.device("cuda"), pin_memory=False
            ):
                pass

        self.assertEqual(seen_pin_memory, [True, False])

    def test_coverage_failure_prevents_inner_post_load(self):
        layer, inner, method = self._create_method()
        layer.quant_method = method
        model = _ToyModel(layer)

        with (
            patch.object(model_loader_module, "is_cuda_alike", return_value=False),
            self.assertRaisesRegex(RuntimeError, "coverage is incomplete"),
        ):
            model_loader_module.DefaultModelLoader.load_weights_and_postprocess(
                model, (), torch.device("cpu")
            )

        self.assertEqual(inner.post_load_calls, 0)

    def test_active_context_wraps_fused_moe_before_weight_creation(self):
        inner = _ToyMethod()
        context = ExpertOffloadContext()
        original_dispatcher = fused_moe_layer_module.create_moe_dispatcher
        fused_moe_layer_module.create_moe_dispatcher = lambda config: object()
        try:
            with (
                get_context().override_server_args(model_path="dummy"),
                get_flags().moe.override(
                    runner_backend=MoeRunnerBackend.AUTO,
                    a2a_backend=MoeA2ABackend.NONE,
                ),
                get_parallel().override(
                    moe_ep_size=1,
                    moe_ep_rank=0,
                    moe_tp_size=1,
                    moe_tp_rank=0,
                    tp_size=1,
                    tp_rank=0,
                ),
                context,
            ):
                layer = FusedMoE(
                    num_experts=2,
                    hidden_size=2,
                    intermediate_size=2,
                    layer_id=0,
                    quant_method=inner,
                )
        finally:
            fused_moe_layer_module.create_moe_dispatcher = original_dispatcher

        self.assertIsInstance(layer.quant_method, OffloadedFusedMoEMethod)
        self.assertIs(layer.quant_method.inner, inner)
        self.assertEqual(len(context.methods), 1)
        self.assertEqual(layer.w13_weight.device.type, "cpu")

    def test_active_context_rejects_fused_or_shared_experts(self):
        context = ExpertOffloadContext()
        original_dispatcher = fused_moe_layer_module.create_moe_dispatcher
        fused_moe_layer_module.create_moe_dispatcher = lambda config: object()
        try:
            with (
                get_context().override_server_args(model_path="dummy"),
                get_flags().moe.override(
                    runner_backend=MoeRunnerBackend.AUTO,
                    a2a_backend=MoeA2ABackend.NONE,
                ),
                get_parallel().override(
                    moe_ep_size=1,
                    moe_ep_rank=0,
                    moe_tp_size=1,
                    moe_tp_rank=0,
                    tp_size=1,
                    tp_rank=0,
                ),
                context,
                self.assertRaisesRegex(RuntimeError, "fused or shared"),
            ):
                FusedMoE(
                    num_experts=3,
                    hidden_size=2,
                    intermediate_size=2,
                    layer_id=0,
                    num_fused_shared_experts=1,
                    quant_method=_ToyMethod(),
                )
        finally:
            fused_moe_layer_module.create_moe_dispatcher = original_dispatcher

    def test_planner_fast_path_and_greedy_ranges(self):
        _, inner, method = self._create_method()
        method.pool = type("Pool", (), {"cache_slots": 4})()
        fast = StandardTopKOutput(
            topk_weights=torch.ones(1, 2),
            topk_ids=torch.tensor([[0, 1]]),
            router_logits=torch.zeros(1, 2),
        )
        with patch.object(torch.Tensor, "to", side_effect=AssertionError("no copy")):
            fast_plan = method.plan_prefill_microbatches(fast)
        self.assertEqual(fast_plan.ranges, [(0, 1)])
        self.assertIsNone(fast_plan.ids_cpu)
        method.pool.cache_slots = 2
        planned = StandardTopKOutput(
            topk_weights=torch.ones(3, 2),
            topk_ids=torch.tensor([[0, 1], [1, 2], [2, 3]]),
            router_logits=torch.zeros(3, 4),
        )
        planned_result = method.plan_prefill_microbatches(planned)
        self.assertEqual(planned_result.ranges, [(0, 1), (1, 2), (2, 3)])
        self.assertTrue(torch.equal(planned_result.ids_cpu, planned.topk_ids))

    def test_cached_view_overrides_runtime_config(self):
        layer, _, method = self._create_method()
        config = type("Config", (), {"num_experts": 3})()
        view = CachedExpertLayerView(
            layer,
            {"w13_weight": torch.ones(3, 2, 2)},
            {
                "num_experts": 3,
                "num_local_experts": 3,
                "moe_runner_config": config,
            },
        )
        self.assertEqual(view.num_experts, 3)
        self.assertEqual(view.num_local_experts, 3)
        self.assertIs(view.moe_runner_config, config)

    def test_bind_runtime_model_matches_quant_method_owner_and_rejects_missing_or_duplicate(
        self,
    ):
        context = ExpertOffloadContext()
        method = OffloadedFusedMoEMethod(_ToyMethod(), context)
        owner = _ToyLayer()
        owner.quant_method = method
        model = nn.Module()
        model.owner = owner
        with patch.object(method, "bind_runtime") as bind:
            context.methods = [method]
            context.bind_runtime_model(model)
            bind.assert_called_once_with(owner)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            context.bind_runtime_model(nn.Module())
        model.other = _ToyLayer()
        model.other.quant_method = method
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            context.bind_runtime_model(model)

    def _runtime_apply_method(self, *, inner_raises=False):
        layer, inner, method = self._create_method()
        calls = []

        class Lease:
            logical_to_physical = {2: 0, 3: 1}

            def release_after(self, stream):
                calls.append(("release", stream))

        class Pool:
            cache_slots = 2

            def acquire(self, ids, transfer, compute):
                calls.append(("acquire", ids, transfer, compute))
                return Lease()

        method.pool = Pool()
        method.runtime_view = layer
        method.context.backend.target_device = torch.device("cuda")
        method.context.backend.transfer_stream = "transfer"

        def apply(view, dispatch):
            calls.append(("apply", dispatch))
            if inner_raises:
                raise RuntimeError("inner failure")
            return "result"

        inner.apply = apply
        topk = StandardTopKOutput(
            topk_weights=torch.ones(2, 2),
            topk_ids=torch.tensor([[2, 3], [3, -1]], dtype=torch.int64),
            router_logits=torch.zeros(2, 4),
        )
        dispatch = StandardDispatchOutput(torch.zeros(2, 2), None, topk)
        return method, dispatch, calls

    def test_apply_remaps_without_mutating_input_and_releases(self):
        method, dispatch, calls = self._runtime_apply_method()
        with patch("torch.cuda.current_stream", return_value="compute"):
            self.assertEqual(method.apply(None, dispatch), "result")
        remapped = next(entry[1] for entry in calls if entry[0] == "apply")
        self.assertTrue(
            torch.equal(dispatch.topk_output.topk_ids, torch.tensor([[2, 3], [3, -1]]))
        )
        self.assertTrue(
            torch.equal(remapped.topk_output.topk_ids, torch.tensor([[0, 1], [1, -1]]))
        )
        self.assertEqual(calls[-1], ("release", "compute"))

    def test_apply_releases_when_inner_raises(self):
        method, dispatch, calls = self._runtime_apply_method(inner_raises=True)
        with patch("torch.cuda.current_stream", return_value="compute"):
            with self.assertRaisesRegex(RuntimeError, "inner failure"):
                method.apply(None, dispatch)
        self.assertEqual(calls[-1], ("release", "compute"))

    def test_apply_releases_when_remapping_raises(self):
        method, dispatch, calls = self._runtime_apply_method()
        with (
            patch("torch.cuda.current_stream", return_value="compute"),
            patch.object(
                torch.Tensor, "clone", side_effect=RuntimeError("remap failure")
            ),
            self.assertRaisesRegex(RuntimeError, "remap failure"),
        ):
            method.apply(None, dispatch)
        self.assertEqual(calls[0][0], "acquire")
        self.assertEqual(calls[-1], ("release", "compute"))

    def test_planned_microbatches_reuse_one_cpu_routing_copy(self):
        """The planner's snapshot is reused rather than copied in each apply."""
        layer, inner, method = self._create_method()
        calls = []

        class Lease:
            def __init__(self, ids):
                self.logical_to_physical = {
                    logical: physical for physical, logical in enumerate(ids)
                }

            def release_after(self, stream):
                calls.append(("release", stream))

        class Pool:
            cache_slots = 2

            def acquire(self, ids, transfer, compute):
                calls.append(("acquire", ids))
                return Lease(ids)

        method.pool = Pool()
        method.runtime_view = layer
        method.context.backend.target_device = torch.device("cuda")
        method.context.backend.transfer_stream = "transfer"
        inner.apply = lambda view, dispatch: dispatch.hidden_states

        fused = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(fused)
        fused.quant_method = method
        fused._dwdp_bound = False
        fused.reduce_results = False
        fused.moe_tp_size = fused.moe_ep_size = 1
        fused.dispatcher = type(
            "Dispatcher",
            (),
            {
                "dispatch": lambda _, hidden_states, topk_output: (
                    StandardDispatchOutput(hidden_states, None, topk_output)
                ),
                "combine": lambda _, combine_input: combine_input,
            },
        )()
        topk = StandardTopKOutput(
            torch.ones(3, 2),
            torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64),
            torch.zeros(3, 4),
        )
        original_to = torch.Tensor.to
        cpu_copy_calls = []

        def observe_to(tensor, *args, **kwargs):
            if kwargs.get("device") == "cpu" and kwargs.get("copy") is True:
                cpu_copy_calls.append(tensor)
            return original_to(tensor, *args, **kwargs)

        with (
            patch.object(torch.Tensor, "to", new=observe_to),
            patch("torch.cuda.current_stream", return_value="compute"),
            patch.object(fused_moe_layer_module, "get_tp_group", return_value=object()),
            patch.object(
                fused_moe_layer_module, "is_allocation_symmetric", return_value=False
            ),
        ):
            output = fused.forward_impl(torch.arange(6.0).reshape(3, 2), topk)

        self.assertEqual(len(cpu_copy_calls), 1)
        self.assertTrue(torch.equal(output, torch.arange(6.0).reshape(3, 2)))
        acquired_ids = [call[1] for call in calls if call[0] == "acquire"]
        self.assertEqual(acquired_ids, [[0, 1], [1, 2], [2, 3]])

    def test_planned_ids_are_cleared_after_a_microbatch_exception(self):
        method, dispatch, _ = self._runtime_apply_method(inner_raises=True)
        fused = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(fused)
        fused.quant_method = method
        fused._dwdp_bound = False
        fused.reduce_results = False
        fused.moe_tp_size = fused.moe_ep_size = 1
        fused.dispatcher = type(
            "Dispatcher",
            (),
            {
                "dispatch": lambda _, hidden_states, topk_output: (
                    StandardDispatchOutput(hidden_states, None, topk_output)
                ),
                "combine": lambda _, combine_input: combine_input,
            },
        )()
        with (
            patch("torch.cuda.current_stream", return_value="compute"),
            patch.object(fused_moe_layer_module, "get_tp_group", return_value=object()),
            patch.object(
                fused_moe_layer_module, "is_allocation_symmetric", return_value=False
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "inner failure"):
                fused.forward_impl(dispatch.hidden_states, dispatch.topk_output)
            self.assertIsNone(method._planned_ids_cpu.get())

            # A following standalone route must take a fresh snapshot rather
            # than consume an ID tensor scoped to the failed block.
            method.inner.apply = lambda view, routed: "recovered"
            self.assertEqual(method.apply(None, dispatch), "recovered")

    def test_apply_all_masked_skips_acquire(self):
        method, dispatch, calls = self._runtime_apply_method()
        masked = dispatch._replace(
            topk_output=dispatch.topk_output._replace(
                topk_ids=torch.full((2, 2), -1, dtype=torch.int64)
            )
        )
        with patch("torch.cuda.current_stream", return_value="compute"):
            method.apply(None, masked)
        self.assertFalse(any(name == "acquire" for name, *_ in calls))

    def test_apply_rejects_non_standard_dispatch_or_topk(self):
        method, dispatch, _ = self._runtime_apply_method()
        with self.assertRaises(NotImplementedError):
            method.apply(None, object())
        with self.assertRaises(NotImplementedError):
            method.apply(None, dispatch._replace(topk_output=object()))

    def test_forward_impl_microbatches_full_dispatch_core_combine_in_order(self):
        layer = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(layer)

        class Method:
            def plan_prefill_microbatches(self, topk):
                return [(0, 1), (1, 3)]

        class StandardFormat:
            def is_standard(self):
                return True

        class DispatchOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states
                self.format = StandardFormat()
                self.hidden_states_scale = None

        class Dispatcher:
            def dispatch(self, hidden_states, topk_output):
                calls.append(("dispatch", hidden_states.clone()))
                return DispatchOutput(hidden_states)

            def combine(self, combine_input):
                calls.append(("combine", combine_input))
                return combine_input

        def run_moe_core(dispatch_output):
            calls.append(("core", dispatch_output))
            return dispatch_output.hidden_states

        layer.quant_method = Method()
        layer._dwdp_bound = False
        layer.reduce_results = False
        layer.moe_tp_size = layer.moe_ep_size = 1
        calls = []
        layer.dispatcher = Dispatcher()
        layer.run_moe_core = run_moe_core
        topk = StandardTopKOutput(
            torch.ones(3, 1), torch.tensor([[0], [1], [2]]), torch.zeros(3, 3)
        )
        with (
            patch.object(fused_moe_layer_module, "get_tp_group", return_value=object()),
            patch.object(
                fused_moe_layer_module, "is_allocation_symmetric", return_value=False
            ),
        ):
            out = layer.forward_impl(torch.arange(6.0).reshape(3, 2), topk)
        self.assertTrue(torch.equal(out, torch.arange(6.0).reshape(3, 2)))
        call_names = [name for name, *_ in calls]
        self.assertEqual(
            call_names,
            ["dispatch", "core", "combine", "dispatch", "core", "combine"],
        )

    def test_forward_impl_microbatch_slices_pre_quant_inputs(self):
        layer = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(layer)

        class Method:
            def plan_prefill_microbatches(self, topk):
                return [(0, 1), (1, 2)]

        class Dispatcher:
            def dispatch(self, hidden_states, topk_output):
                return StandardDispatchOutput(hidden_states, None, topk_output)

            def combine(self, combine_input):
                return combine_input

        def run_moe_core(dispatch_output):
            captured.append(dispatch_output.hidden_states_pre_quant)
            return dispatch_output.hidden_states

        layer.quant_method = Method()
        layer._dwdp_bound = False
        layer.reduce_results = False
        layer.moe_tp_size = layer.moe_ep_size = 1
        captured = []
        layer.dispatcher = Dispatcher()
        layer.run_moe_core = run_moe_core
        topk = StandardTopKOutput(
            torch.ones(2, 1), torch.tensor([[0], [1]]), torch.zeros(2, 2)
        )
        prequant = (torch.arange(4.0).reshape(2, 2), torch.arange(2.0).reshape(2, 1))
        with (
            patch.object(fused_moe_layer_module, "get_tp_group", return_value=object()),
            patch.object(
                fused_moe_layer_module, "is_allocation_symmetric", return_value=False
            ),
        ):
            layer.forward_impl(torch.zeros(2, 2), topk, pre_quant_input=prequant)
        self.assertEqual([item[0].shape[0] for item in captured], [1, 1])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_cache_copy_and_remap_smoke(self):
        source = torch.tensor([[1.0], [2.0]])
        cached = source[1:2].to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        self.assertEqual(cached.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
