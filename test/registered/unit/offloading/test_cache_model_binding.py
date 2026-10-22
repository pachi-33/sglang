"""Native layout and runner contracts with CPU stand-ins only for NPU kernels."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_cache_runtime import batch
from test_npu_backend import Assembly

from sglang.srt.offloading.moe.config import Phase, fixed_waves
from sglang.srt.offloading.moe.npu import experts
from sglang.srt.offloading.moe.owner import MoEOffloadOwner, OwnerState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class CpuCore:
    """Test FFN: scale each expert's rows by its toy weight, leave padding NaN."""

    def run(self, inputs, quant, _):
        output = torch.full_like(inputs.hidden_states, float("nan"))
        offset = 0
        for expert, count in enumerate(inputs.expert_tokens.tolist()):
            output[offset : offset + count] = (
                inputs.hidden_states[offset : offset + count]
                * quant.w13_weight[expert, 0]
            )
            offset += count
        return SimpleNamespace(hidden_states=output)


class TestNativeCacheLayout(unittest.TestCase):
    def setUp(self):
        self.assembly = Assembly(layer_count=1, local_experts=4)
        self.runtime = self.assembly.finalize()
        self.context = self.runtime._new_context(Phase.DECODE, capturing=True)
        self.runtime._context = self.context
        self.addCleanup(self.assembly.close)
        self.scope = patch.object(
            experts, "_compute_scope", lambda *args: nullcontext()
        )
        self.scope.start()
        self.addCleanup(self.scope.stop)
        self.adapter = experts.NPUDispatchLayout(
            4, 2, 8, "cpu", self.runtime.ops.error_status
        )

    def received(self, counts, rows=6, fmt="deepep_ll"):
        return SimpleNamespace(
            raw=SimpleNamespace(
                hidden_states=torch.arange(rows * 2, dtype=torch.bfloat16).reshape(
                    rows, 2
                ),
                hidden_states_scale=torch.arange(rows, dtype=torch.float32).unsqueeze(
                    1
                ),
                masked_m=torch.tensor(counts),
                num_recv_tokens_per_expert=counts,
                format=fmt,
            )
        )

    def test_live_counts_padding_and_scale_rows_without_host_reads(self):
        address = self.adapter._owned["output"].data_ptr()
        for counts in ([0, 2, 1, 0], [1, 0, 0, 3], [0, 0, 0, 0]):
            raw = self.received(counts)
            with ExitStack() as stack:
                for method in ("item", "tolist", "cpu", "__bool__"):
                    stack.enter_context(
                        patch.object(
                            torch.Tensor, method, side_effect=AssertionError(method)
                        )
                    )
                layout = self.adapter.bind(raw, self.context)
                for wave in fixed_waves(4, 2):
                    selected = self.adapter.select_wave(layout, wave, self.context)
                    self.adapter.write_wave(
                        layout, wave, selected.hidden_states * 2, self.context
                    )
            count = sum(counts)
            torch.testing.assert_close(
                layout.output[:count], raw.raw.hidden_states[:count] * 2
            )
            self.assertEqual(int(layout.output[count:].count_nonzero()), 0)
            self.assertEqual(layout.output.data_ptr(), address)
        raw = self.received([0, 2, 1, 0])
        layout = self.adapter.bind(raw, self.context)
        for wave in fixed_waves(4, 2):
            selected = self.adapter.select_wave(layout, wave, self.context)
            offset, count = int(selected.offset), int(selected.row_count)
            torch.testing.assert_close(
                selected.hidden_states_scale[:count],
                raw.raw.hidden_states_scale[offset : offset + count],
            )
            self.adapter.write_wave(layout, wave, selected.hidden_states, self.context)

    def test_normal_dispatch_counts_are_prefill_only(self):
        raw = self.received([1, 1, 0, 0], fmt="deepep_normal")
        with self.assertRaisesRegex(ValueError, "Decode requires"):
            self.adapter.bind(raw, self.context)
        self.context.phase, self.context.capturing = Phase.PREFILL, False
        layout = self.adapter.bind(raw, self.context)
        self.assertEqual(layout.counts.tolist(), [1, 1, 0, 0])

    def test_byte_gather_preserves_bits_with_npu_mask_dtype_constraint(self):
        adapter = experts.NPUDispatchLayout(
            4, 256, 8, "cpu", self.runtime.ops.error_status
        )
        # All byte patterns, including high bits and floating-point NaNs.
        payload = torch.arange(256).to(torch.uint8).reshape(8, 32)
        indices = torch.tensor([7, 2, 5, 0, 6, 3, 1, 4])
        masked_fill = torch.Tensor.masked_fill_

        def npu_masked_fill(tensor, mask, value):
            if tensor.dtype == torch.uint8:
                raise RuntimeError("NPU masked_fill_ does not support UINT8")
            return masked_fill(tensor, mask, value)

        for dtype in (
            torch.uint8,
            torch.int8,
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float8_e4m3fn,
        ):
            source = payload.view(dtype)
            for name in ("input_bytes", "scale_bytes"):
                for count in (8, 3, 0):
                    with self.subTest(dtype=dtype, buffer=name, count=count):
                        expected = payload.index_select(0, indices)
                        expected[count:].zero_()
                        with patch.object(
                            torch.Tensor, "masked_fill_", npu_masked_fill
                        ):
                            output = adapter._gather_bytes(
                                source, indices, torch.tensor(count), name
                            )
                        torch.testing.assert_close(output.view(torch.uint8), expected)
                        self.assertEqual(output.dtype, dtype)
                        self.assertEqual(output.shape, source.shape)
                        self.assertEqual(
                            output.data_ptr(), adapter._owned[name].data_ptr()
                        )
        torch.testing.assert_close(
            payload, torch.arange(256).to(torch.uint8).reshape(8, 32)
        )

    def test_zero_receive_rows_preserve_geometry(self):
        layout = self.adapter.bind(self.received([0] * 4, rows=0), self.context)
        for wave in fixed_waves(4, 2):
            selected = self.adapter.select_wave(layout, wave, self.context)
            self.adapter.write_wave(layout, wave, selected.hidden_states, self.context)
        self.assertEqual(layout.output.shape, (0, 2))
        self.assertEqual(layout.output.dtype, torch.bfloat16)

    def test_native_runner_masks_unwritten_padding_and_uses_bank_views(self):
        bank = self.assembly.backend.format.banks[1]
        bank.tensors["w13_weight"][0].fill_(2)
        bank.tensors["w13_weight"][1].fill_(3)
        valid = torch.tensor([True, True, False, False])
        native_bank = experts.NPUExpertBank(
            bank.quant_info(), valid, self.runtime.ops.error_status
        )
        quant = native_bank.for_width(2)
        self.assertEqual(
            quant.w13_weight.data_ptr(), bank.tensors["w13_weight"].data_ptr()
        )
        layout = self.adapter.bind(self.received([1, 2, 0, 0]), self.context)
        selected = self.adapter.select_wave(layout, fixed_waves(4, 2)[0], self.context)
        runner = experts.NPUExpertRunner(
            None, _core=CpuCore(), _input_type=SimpleNamespace
        )
        result = runner(
            3, selected, SimpleNamespace(expert_bank=native_bank), self.context
        )
        torch.testing.assert_close(
            result.output[:3],
            selected.hidden_states[:3] * torch.tensor([2, 3, 3])[:, None],
        )
        self.assertEqual(int(result.output[3:].count_nonzero()), 0)
        self.assertTrue(result.read_done.completion.done())

    def test_invalid_bank_is_not_used_and_fault_masks_earlier_waves(self):
        raw = self.received([1, 1, 0, 0])
        layout = self.adapter.bind(raw, self.context)
        selected = self.adapter.select_wave(layout, fixed_waves(4, 2)[0], self.context)
        bank = self.assembly.backend.format.banks[1]
        binding = SimpleNamespace(
            expert_bank=experts.NPUExpertBank(
                bank.quant_info(),
                torch.zeros(4, dtype=torch.bool),
                self.runtime.ops.error_status,
            )
        )
        runner = experts.NPUExpertRunner(
            None, _core=CpuCore(), _input_type=SimpleNamespace
        )
        result = runner(3, selected, binding, self.context)
        self.assertEqual(int(self.runtime.ops.error_status), 128)
        self.assertEqual(int(result.output.count_nonzero()), 0)
        self.adapter.write_wave(layout, selected.wave, result.output, self.context)
        layout.output.fill_(9)
        self.assertEqual(
            int(
                self.adapter.combine_input(
                    layout, self.context
                ).hidden_states.count_nonzero()
            ),
            0,
        )


class TestNativeCacheFaultPropagation(unittest.TestCase):
    def setUp(self):
        self.assembly = Assembly(layer_count=1, local_experts=4)
        self.runtime = self.assembly.finalize()
        self.owner = MoEOffloadOwner(
            self.assembly.config, self.assembly.catalog, self.runtime
        )
        self.addCleanup(self.owner.close)
        scope = patch.object(experts, "_compute_scope", lambda *args: nullcontext())
        scope.start()
        self.addCleanup(scope.stop)
        self.adapter = experts.NPUDispatchLayout(
            4, 2, 6, "cpu", self.runtime.ops.error_status
        )

    def received(self, counts):
        return SimpleNamespace(
            raw=SimpleNamespace(
                hidden_states=torch.ones(6, 2, dtype=torch.bfloat16),
                hidden_states_scale=None,
                masked_m=torch.tensor(counts, dtype=torch.int64),
                format="deepep_ll",
            )
        )

    def assert_bad_counts_fail(self, decode):
        # Trigger the actual layout fault; do not inject the error flag directly.
        with (
            self.assertLogs("sglang.srt.offloading.moe.npu.runtime", level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "device fault: status=64"),
            patch.object(
                self.owner, "finish_forward", wraps=self.owner.finish_forward
            ) as finish,
        ):
            with self.runtime.forward_scope(self.owner, batch(decode)) as context:
                layout = self.adapter.bind(self.received([7, 0, 0, 0]), context)
                layout.output.fill_(9)
                output = self.adapter.combine_input(layout, context).hidden_states
                self.assertEqual(output.count_nonzero().item(), 0)
        finish.assert_not_called()
        self.assertEqual(self.owner.state, OwnerState.FAILED)
        # A valid subsequent batch must not run with the sticky fault latch.
        with self.assertRaisesRegex(RuntimeError, "owner is failed"):
            with self.runtime.forward_scope(self.owner, batch(decode)):
                self.fail("A failed runtime accepted another forward")

    def test_decode_layout_fault_fails_before_return_and_blocks_next_batch(self):
        self.assert_bad_counts_fail(decode=True)

    def test_prefill_layout_fault_fails_before_return_and_blocks_next_batch(self):
        self.assert_bad_counts_fail(decode=False)

    def test_missing_expert_weights_fail_instead_of_returning_zero_output(self):
        with (
            self.assertLogs("sglang.srt.offloading.moe.npu.runtime", level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "device fault: status=128"),
        ):
            with self.runtime.forward_scope(self.owner, batch(True)) as context:
                layout = self.adapter.bind(self.received([1, 1, 0, 0]), context)
                selected = self.adapter.select_wave(
                    layout, fixed_waves(4, 2)[0], context
                )
                bank = self.assembly.backend.format.banks[1]
                binding = SimpleNamespace(
                    expert_bank=experts.NPUExpertBank(
                        bank.quant_info(),
                        torch.zeros(4, dtype=torch.bool),
                        self.runtime.ops.error_status,
                    )
                )
                runner = experts.NPUExpertRunner(
                    None, _core=CpuCore(), _input_type=SimpleNamespace
                )
                result = runner(3, selected, binding, context)
                self.assertEqual(result.output.count_nonzero().item(), 0)
        self.assertEqual(self.owner.state, OwnerState.FAILED)

    def test_healthy_replays_preserve_outputs_without_device_wide_sync(self):
        with patch.object(
            self.runtime.device_runtime,
            "synchronize",
            side_effect=AssertionError("Unexpected device-wide synchronization"),
        ):
            for _ in range(2):
                with self.runtime.forward_scope(self.owner, batch(True)) as context:
                    layout = self.adapter.bind(self.received([1, 1, 0, 0]), context)
                    layout.output.fill_(9)
                    output = self.adapter.combine_input(layout, context).hidden_states
                torch.testing.assert_close(output, torch.full_like(output, 9))
                self.assertEqual(self.owner.state, OwnerState.READY)


if __name__ == "__main__":
    unittest.main()
