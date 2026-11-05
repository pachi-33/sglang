"""Mixed offload decode masks across attention TP shards; no NPU required."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.token_dispatcher import ascend_ops as dispatcher_module
from sglang.srt.layers.moe.token_dispatcher.ascend_ops import AscendOpsDispatcher
from sglang.srt.model_executor import forward_batch_info as batch_module
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.offloading.moe.npu import ascend_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAscendDecodeMask(CustomTestCase):
    def _owner(self, *, dp=8, cp=1, capacity=128, disable_gather=False):
        owner = object.__new__(ascend_ops.AscendOpsOffloader)
        owner.cfg = {"decode_mask_capacity": capacity}
        owner.device = torch.device("cpu")
        with patch.object(
            ascend_ops,
            "get_parallel",
            return_value=SimpleNamespace(
                tp_size=16,
                moe_ep_size=16,
                attn_dp_size=dp,
                attn_cp_size=cp,
                disable_attn_tp_gather=disable_gather,
            ),
        ):
            owner._init_decode_mask()
        return owner

    @staticmethod
    def _context(bound):
        return forward_context(
            ForwardContext(attn_backend=None, moe_num_token_non_padded=lambda: bound)
        )

    def _local_count(self, real, bucket, width, rank, *, sharded=True):
        with patch.object(
            batch_module,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=width, attn_tp_rank=rank),
        ):
            return batch_module.compute_local_num_token_non_padded(
                torch.tensor(real, dtype=torch.int32), bucket, sharded=sharded
            )

    def test_uneven_dp_batches_and_trailing_zero_shards(self):
        # TP=EP=16, attention DP8/4 gives attention TP2/4. The bucket, not
        # ceil(real / attention TP), determines where each shard starts.
        for dp, bucket, real, expected in (
            (8, 8, 6, [4, 2]),
            (8, 8, 1, [1, 0]),
            (4, 16, 10, [4, 4, 2, 0]),
            (4, 16, 0, [0, 0, 0, 0]),
            (16, 16, 6, [6]),
        ):
            owner = self._owner(dp=dp)
            for rank, valid in enumerate(expected):
                with self.subTest(dp=dp, real=real, rank=rank):
                    bound = self._local_count(real, bucket, len(expected), rank)
                    with self._context(bound):
                        mask = owner.decode_row_mask(128)
                    self.assertEqual(
                        mask.tolist(), [True] * valid + [False] * (128 - valid)
                    )

    def test_replicated_input_keeps_the_full_valid_prefix(self):
        owner = self._owner(dp=1)
        for rank in range(4):
            bound = self._local_count(6, 8, 4, rank, sharded=False)
            with self._context(bound):
                mask = owner.decode_row_mask(128)
            self.assertEqual(mask.sum().item(), 6)

    def test_captured_comparison_refreshes_without_python_or_new_storage(self):
        owner = self._owner()
        address = owner.decode_active_rows.data_ptr()
        # Trace the production tensor operations as a CPU proxy for capture.
        # Each bucket retains the same mask, but has different shard offsets.
        for bucket, cases in (
            (8, ((6, 2), (0, 0), (8, 4), (1, 0))),
            (16, ((6, 0), (10, 2), (16, 8), (0, 0))),
        ):

            def run(real):
                with patch.object(
                    batch_module,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_tp_size=2, attn_tp_rank=1),
                ):
                    bound = batch_module.compute_local_num_token_non_padded(
                        real, bucket, sharded=True
                    )
                with self._context(bound):
                    return owner.decode_row_mask(128)

            graph = torch.jit.trace(
                run, (torch.tensor(6, dtype=torch.int32),), check_trace=False
            )
            # With no active ForwardContext, only captured tensor operations
            # can update the mask. This also catches frozen Python/.item counts.
            for real, valid in cases:
                mask = graph(torch.tensor(real, dtype=torch.int32))
                self.assertEqual(mask.data_ptr(), address)
                self.assertEqual(
                    mask.tolist(), [True] * valid + [False] * (128 - valid)
                )

    def test_mtp_counts_candidate_rows_and_clears_the_idle_tail(self):
        owner = self._owner()
        bound = torch.tensor(6, dtype=torch.int32)
        with self._context(bound):
            self.assertEqual(owner.decode_row_mask(128).sum().item(), 6)
            bound.fill_(1)
            self.assertEqual(owner.decode_row_mask(128).sum().item(), 1)
            bound.zero_()
            self.assertFalse(owner.decode_row_mask(128).any().item())

    def test_variable_dispatch_widths_reuse_and_refresh_the_same_buffer(self):
        owner = self._owner()
        address = owner.decode_active_rows.data_ptr()
        for width, valid in ((16, 10), (4, 1), (8, 6), (16, 0)):
            with self._context(torch.tensor(valid, dtype=torch.int32)):
                mask = owner.decode_row_mask(width)
            self.assertEqual(mask.data_ptr(), address)
            self.assertEqual(mask.tolist(), [True] * valid + [False] * (width - valid))

    def test_missing_metadata_capacity_and_cp_fail_explicitly(self):
        owner = self._owner(capacity=8)
        with self.assertRaisesRegex(RuntimeError, "MoE-local"):
            owner.decode_row_mask(8)
        with self._context(None), self.assertRaisesRegex(RuntimeError, "MoE-local"):
            owner.decode_row_mask(8)
        with self.assertRaisesRegex(ValueError, "decode_mask_capacity"):
            owner.decode_row_mask(9)
        with self.assertRaisesRegex(ValueError, "CP=1"):
            self._owner(cp=2)
        with self.assertRaisesRegex(ValueError, "remove --disable-attn-tp-gather"):
            self._owner(disable_gather=True)
        self._owner(dp=16, disable_gather=True)
        for capacity in (0, -1, True):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(
                ValueError, "positive"
            ):
                self._owner(capacity=capacity)

    def test_mc2_dispatch_and_combine_share_the_local_mask(self):
        owner = self._owner()
        dispatcher = object.__new__(AscendOpsDispatcher)
        dispatcher._offload_layer = SimpleNamespace(_ascend_ops_offloader=owner)
        dispatcher._pad_buffers = {}
        dispatcher._quant_mode = Mock(return_value=0)
        dispatcher._get_ep_comm_name = Mock(return_value="ep")
        dispatcher.ep_size, dispatcher.ep_rank, dispatcher.num_experts = 16, 1, 16
        dispatcher.moe_ascend_ops_config = SimpleNamespace(
            ll_dispatch_alg="", ll_combine_alg=""
        )
        sent = []

        def dispatch(x, ids, *args, x_active_mask, **kwargs):
            sent.append(x_active_mask.clone())
            valid = x_active_mask.nonzero().flatten()
            # A rank with no source rows still receives expert work from peers.
            received = x[valid] if valid.numel() else torch.tensor([[9.0]])
            return (
                received,
                None,
                valid,
                torch.tensor([received.shape[0]]),
                torch.empty(0),
                None,
                None,
            )

        def combine(x, ids, valid, counts, weights, *args, x_active_mask, **kwargs):
            self.assertEqual(
                x_active_mask.data_ptr(), owner.decode_active_rows.data_ptr()
            )
            result = torch.zeros((128, 1))
            result[valid] = x[: valid.numel()]
            return result

        ops = SimpleNamespace(
            npu_moe_distribute_dispatch_v2=Mock(side_effect=dispatch),
            npu_moe_distribute_combine_v2=Mock(side_effect=combine),
        )
        with patch.object(dispatcher_module, "torch_npu", ops, create=True):
            for valid in (2, 0):
                with self._context(torch.tensor(valid, dtype=torch.int32)):
                    output = dispatcher._dispatch_mc2(
                        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                        torch.zeros((4, 1), dtype=torch.int32),
                        torch.ones((4, 1)),
                        padded_bs=128,
                        dtype_mode="low_latency",
                    )
                    self.assertEqual(output.expert_tokens.sum().item(), valid or 1)
                    result = dispatcher._combine_mc2(
                        output.hidden_states * 2, output.forward_state
                    )
                self.assertEqual(
                    result.flatten().tolist(),
                    [2.0, 4.0, 0.0, 0.0] if valid else [0.0] * 4,
                )
                self.assertEqual(sent[-1].sum().item(), valid)
        self.assertEqual(ops.npu_moe_distribute_dispatch_v2.call_count, 2)
        self.assertEqual(ops.npu_moe_distribute_combine_v2.call_count, 2)


if __name__ == "__main__":
    unittest.main()
