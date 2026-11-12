"""CPU contracts for large prefill pools; device kernels need NPU validation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.moe_runner import ascend
from sglang.srt.layers.moe.token_dispatcher import ascend_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSplitPrefillGmm(CustomTestCase):
    def _runner(self, resident_experts=2):
        calls = []
        weights = {
            "w13": torch.arange(32, dtype=torch.float32).reshape(4, 2, 4) / 32,
            "w2": torch.arange(16, dtype=torch.float32).reshape(4, 2, 2) / 16,
        }
        resident = SimpleNamespace(
            name="resident", **{k: v[:resident_experts] for k, v in weights.items()}
        )
        cold = SimpleNamespace(
            name="cold", **{k: v[resident_experts:] for k, v in weights.items()}
        )

        def project(
            quant,
            x,
            counts,
            *,
            pertoken_scale,
            output_dtype,
            weight_prefix,
            group_list_type,
        ):
            self.assertEqual(group_list_type, 1)
            counts = counts.tolist()
            self.assertEqual(sum(counts), x.shape[0])
            self.assertEqual(len(counts), getattr(quant, weight_prefix).shape[0])
            calls.append((quant.name, weight_prefix, counts))
            x = x.float()
            if pertoken_scale is not None:
                self.assertEqual(pertoken_scale.shape, (x.shape[0],))
                x = x * pertoken_scale[:, None]
            results, offset = [], 0
            for count, weight in zip(counts, getattr(quant, weight_prefix)):
                results.append(x[offset : offset + count] @ weight)
                offset += count
            return torch.cat(results).to(output_dtype)

        class Activation(ascend.NPUSwigluDeepEPKernel):
            def _apply_activation(inner, x, *, group_list, group_list_type):
                self.assertEqual(group_list_type, 1)
                self.assertEqual(sum(group_list.tolist()), x.shape[0])
                gate, up = x.chunk(2, dim=-1)
                return torch.nn.functional.silu(gate) * up, None

        owner = SimpleNamespace(
            prepare_weights=Mock(return_value=resident),
            wait_for_w2=Mock(side_effect=lambda layer: calls.append("cold-w2-ready")),
            prefill=SimpleNamespace(
                split_gmm=True,
                resident_experts=resident_experts,
                cold_weights=Mock(
                    side_effect=lambda layer: (calls.append("cold-w13-ready"), cold)[1]
                ),
                finish_layer=Mock(),
            ),
        )
        core = object.__new__(ascend.AscendRunnerCore)
        core.config = SimpleNamespace(
            swiglu_limit=None,
            activation="silu",
            layer=SimpleNamespace(
                layer_id=9,
                _ascend_ops_offloader=owner,
                w13_kernel=SimpleNamespace(apply=project),
                w2_kernel=SimpleNamespace(apply=project),
            ),
        )
        core.activation = object.__new__(Activation)
        return core, owner, calls, weights

    def test_split_preserves_weight_rows_scales_and_output_order(self):
        for counts in ([1, 0, 2, 1], [1, 2, 0, 0], [0, 0, 1, 2]):
            for dtype in (torch.int8, torch.bfloat16, torch.float16):
                with self.subTest(counts=counts, dtype=dtype):
                    core, owner, calls, weights = self._runner()
                    rows = sum(counts)
                    x = (torch.arange(rows * 2).reshape(rows, 2) - 2).to(dtype)
                    scale = torch.arange(1, rows + 1, dtype=torch.float32) / 8
                    inputs = ascend.AscendRunnerInput(
                        x,
                        scale[:, None] if dtype == torch.int8 else None,
                        torch.tensor(counts),
                        1,
                        sum(counts[:2]),
                    )
                    with (
                        patch.object(
                            ascend, "get_is_extend_in_batch", return_value=True
                        ),
                        patch.object(
                            torch.Tensor, "cpu", side_effect=AssertionError("D2H")
                        ),
                        patch.object(
                            torch.Tensor, "item", side_effect=AssertionError("sync")
                        ),
                    ):
                        result = core.run(inputs, None, {}).hidden_states

                    # Independent per-expert reference, with the same rounding
                    # at the projection boundary; no physical-slot selection.
                    expected, offset = [], 0
                    output_dtype = (
                        torch.float16 if dtype == torch.float16 else torch.bfloat16
                    )
                    for expert, count in enumerate(counts):
                        value = x[offset : offset + count].float()
                        if dtype == torch.int8:
                            value *= scale[offset : offset + count, None]
                        projected = (value @ weights["w13"][expert]).to(output_dtype)
                        gate, up = projected.chunk(2, dim=-1)
                        activated = torch.nn.functional.silu(gate) * up
                        expected.append(
                            (activated.float() @ weights["w2"][expert]).to(output_dtype)
                        )
                        offset += count
                    torch.testing.assert_close(
                        result, torch.cat(expected), rtol=0, atol=0
                    )
                    sequence = []
                    if sum(counts[:2]):
                        sequence += [
                            ("resident", "w13", counts[:2]),
                            ("resident", "w2", counts[:2]),
                        ]
                    if sum(counts[2:]):
                        sequence += [
                            "cold-w13-ready",
                            ("cold", "w13", counts[2:]),
                            "cold-w2-ready",
                            ("cold", "w2", counts[2:]),
                        ]
                    self.assertEqual(calls, sequence)
                    owner.prefill.finish_layer.assert_not_called()

    def test_single_expert_in_either_partition_preserves_full_result(self):
        results = []
        for resident in (1, 2, 3):
            core, owner, calls, _ = self._runner(resident)
            inputs = ascend.AscendRunnerInput(
                torch.ones((4, 2), dtype=torch.bfloat16),
                None,
                torch.ones(4, dtype=torch.int64),
                1,
                resident,
            )
            with patch.object(ascend, "get_is_extend_in_batch", return_value=True):
                results.append(core.run(inputs, None, {}).hidden_states)
            self.assertEqual(owner.prefill.cold_weights.call_count, 1)
            self.assertEqual(calls[0], ("resident", "w13", [1] * resident))
            self.assertEqual(calls[-1], ("cold", "w2", [1] * (4 - resident)))
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=0)

    def test_non_offloaded_fused_gmm_keeps_original_counts_and_scale(self):
        core = object.__new__(ascend.AscendRunnerCore)
        x = torch.ones((3, 2), dtype=torch.bfloat16)
        scale = torch.ones(3)
        quant = object()
        fused = Mock(return_value=(x * 2, scale))
        down = Mock(return_value=x * 6)
        core.config = SimpleNamespace(
            layer=SimpleNamespace(
                w13_kernel=SimpleNamespace(apply_fused_gmm1_swiglu=fused),
                w2_kernel=SimpleNamespace(apply=down),
            )
        )
        core.activation = Mock()
        for kind, counts in ((0, [1, 3]), (1, [1, 2])):
            group_list = torch.tensor(counts)
            inputs = ascend.AscendRunnerInput(x, None, group_list, kind)
            with patch.object(ascend, "_uses_fused_gmm1", return_value=True):
                result = core.run(inputs, quant, {})
            torch.testing.assert_close(result.hidden_states, x * 6)
            self.assertIs(fused.call_args.args[0], quant)
            self.assertIs(fused.call_args.args[2], group_list)
            self.assertEqual(fused.call_args.kwargs["group_list_type"], kind)
            self.assertIs(down.call_args.args[2], group_list)
            self.assertIs(down.call_args.kwargs["pertoken_scale"], scale)
            self.assertEqual(down.call_args.kwargs["group_list_type"], kind)
        core.activation._apply_activation.assert_not_called()

    def test_empty_and_deepep_dummy_receives_skip_both_ffns(self):
        for dummy in (False, True):
            core, owner, calls, _ = self._runner()
            inputs = ascend.AscendRunnerInput(
                torch.ones((int(dummy), 2), dtype=torch.int8),
                None,
                torch.zeros(4, dtype=torch.int64),
                1,
                0,
            )
            with patch.object(ascend, "get_is_extend_in_batch", return_value=True):
                result = core.run(inputs, None, {"deepep_empty_receive": dummy})
            self.assertEqual(result.hidden_states.shape, inputs.hidden_states.shape)
            self.assertEqual(result.hidden_states.dtype, torch.bfloat16)
            self.assertEqual(result.hidden_states.count_nonzero(), 0)
            self.assertEqual(calls, [])
            owner.prefill.finish_layer.assert_not_called()

    def test_missing_or_invalid_boundary_fails_before_ffn(self):
        for boundary in (None, -1, 4, True):
            core, _, calls, _ = self._runner()
            inputs = ascend.AscendRunnerInput(
                torch.ones((3, 2), dtype=torch.int8),
                torch.ones(3),
                torch.tensor([1, 0, 2, 0]),
                1,
                boundary,
            )
            with (
                patch.object(ascend, "get_is_extend_in_batch", return_value=True),
                self.assertRaisesRegex(ValueError, "resident row count"),
            ):
                core.run(inputs, None, {})
            self.assertEqual(calls, [])

    def test_deepep_uses_existing_host_counts_only_in_split_mode(self):
        for split in (False, True):
            config = SimpleNamespace(
                num_local_experts=4,
                layer=SimpleNamespace(
                    _ascend_ops_offloader=SimpleNamespace(
                        prefill=SimpleNamespace(split_gmm=split, resident_experts=2)
                    )
                ),
            )
            output = (
                torch.ones((3, 2), dtype=torch.int8),
                torch.ones(3),
                None,
                None,
                [1, 0, 2, 0],
            )
            with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")):
                inputs = ascend.pre_permute_deepep_normal_to_ascend(
                    output, None, config, {}
                )
            self.assertEqual(inputs.prefill_resident_rows, 1 if split else None)
            self.assertEqual(inputs.expert_tokens.tolist(), [1, 0, 2, 0])

    def test_ascend_ops_reuses_count_readback_and_combines_once(self):
        # Receive on rank 1, from two source ranks, in source-major order.
        counts = torch.tensor([[0, 2, 1, 0], [1, 0, 0, 2]])
        wire = torch.tensor(
            [[2, 4], [3, 5], [4, 6], [1, 3], [5, 7], [6, 8]], dtype=torch.bfloat16
        )
        order = torch.tensor([3, 0, 1, 2, 4, 5])
        for split in (False, True):
            core, owner, _, _ = self._runner()
            owner.prefill.split_gmm = split
            dispatcher = object.__new__(ascend_ops.AscendOpsDispatcher)
            dispatcher._offload_layer = core.config.layer
            dispatcher.num_experts, dispatcher.num_local_experts = 8, 4
            dispatcher.hidden_size = 2
            readbacks = []
            original_cpu = torch.Tensor.cpu

            def readback(tensor, *args, **kwargs):
                readbacks.append(tuple(tensor.shape))
                return original_cpu(tensor, *args, **kwargs)

            def gather(dst, src, **kwargs):
                dst.zero_()
                dst.view(2, 8)[:, 4:] = counts

            def routing(x, ids, experts, **kwargs):
                ids = ids.reshape(-1)
                perm = torch.argsort(ids, stable=True)
                return SimpleNamespace(
                    hidden_states=x.repeat_interleave(ids.numel() // x.shape[0], dim=0)[
                        perm
                    ],
                    scale=None,
                    expanded_row_idx=torch.argsort(perm),
                )

            with (
                patch.object(dispatcher, "_quant_mode", return_value=-1),
                patch.object(ascend_ops.dist, "get_world_size", return_value=2),
                patch.object(ascend_ops.dist, "get_rank", return_value=1),
                patch.object(
                    ascend_ops.dist, "all_gather_into_tensor", side_effect=gather
                ) as gather_call,
                patch.object(
                    ascend_ops.dist,
                    "all_to_all_single",
                    side_effect=lambda dst, src, **kw: dst.copy_(wire),
                ) as exchange,
                patch.object(ascend_ops, "npu_moe_init_routing", side_effect=routing),
                patch.object(ascend, "npu_moe_init_routing", side_effect=routing),
                patch.object(torch.Tensor, "cpu", readback),
            ):
                output = dispatcher._dispatch_alltoallv(
                    torch.ones((2, 2), dtype=torch.bfloat16),
                    torch.ones((2, 2)),
                    torch.tensor([[0, 4], [7, 7]]),
                    object(),
                )
                config = SimpleNamespace(num_local_experts=4)
                state = {}
                inputs = ascend._pre_permute_alltoallv(output, config, state)
            self.assertEqual(gather_call.call_count, 1)
            self.assertEqual(exchange.call_count, 1)
            self.assertEqual(len(readbacks), 2)
            self.assertEqual(readbacks[-1], (2, 4) if split else (2,))
            self.assertEqual(output.forward_state.input_splits, [1, 3])
            self.assertEqual(output.forward_state.output_splits, [3, 3])
            self.assertEqual(inputs.prefill_resident_rows, 3 if split else None)
            self.assertEqual(inputs.expert_tokens.tolist(), [1, 2, 1, 2])
            torch.testing.assert_close(inputs.hidden_states, wire[order])

            if split:
                with patch.object(ascend, "get_is_extend_in_batch", return_value=True):
                    result = core.run(inputs, None, {})
                with patch.object(
                    ascend,
                    "npu_moe_finalize_routing",
                    side_effect=lambda x, inv: x[inv],
                ):
                    combined = ascend.post_permute_ascend_to_ascend_ops(
                        result, None, config, state
                    )
                torch.testing.assert_close(
                    combined.hidden_states[order], result.hidden_states
                )
                with patch.object(
                    dispatcher, "_combine_alltoallv", return_value="done"
                ) as combine:
                    self.assertEqual(dispatcher.combine(combined), "done")
                combine.assert_called_once_with(
                    combined.hidden_states, output.forward_state
                )
                owner.prefill.finish_layer.assert_called_once_with(9)


if __name__ == "__main__":
    unittest.main()
