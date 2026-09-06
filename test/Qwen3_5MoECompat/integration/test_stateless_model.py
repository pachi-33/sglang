import json
import os
import unittest
from pathlib import Path
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase

import torch

from sglang.srt.layers.qwen3_5.runner import Qwen35StatelessRunner
from sglang.srt.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

MODEL_DIR = Path(
    os.environ.get(
        "QWEN35_MODEL_DIR",
        "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
    )
)


class TestStatelessFourLayerSlice(V100TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not MODEL_DIR.is_dir():
            raise unittest.SkipTest("real checkpoint unavailable")
        cls.model = Qwen3_5MoeForConditionalGeneration.from_checkpoint(
            MODEL_DIR, selected_layer_ids=range(4)
        )

    @classmethod
    def tearDownClass(cls):
        # The layer-39 smoke below intentionally owns a separate FP16-expert
        # resident set; release this four-layer model first.
        del cls.model
        torch.cuda.empty_cache()
        super().tearDownClass()

    @staticmethod
    def _packed(lengths, *, start_id=1):
        tokens = sum(lengths)
        cu = [0]
        positions = []
        for length in lengths:
            cu.append(cu[-1] + length)
            positions.extend(range(length))
        return (
            torch.arange(start_id, start_id + tokens, device="cuda", dtype=torch.int32),
            torch.tensor(positions, device="cuda", dtype=torch.int32),
            torch.tensor(cu, device="cuda", dtype=torch.int32),
        )

    def _forward(self, lengths, *, logits_indices=None, start_id=1):
        ids, positions, cu = self._packed(lengths, start_id=start_id)
        return self.model.forward_no_cache(
            input_ids=ids,
            positions=positions,
            cu_seqlens=cu,
            max_seqlen=max(lengths, default=0),
            logits_indices=logits_indices,
        )

    @staticmethod
    def _nrmse(actual, expected):
        return (
            (actual.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-8)
        ).item()

    @staticmethod
    def _report(**values):
        print("M4_METRIC=" + json.dumps(values, sort_keys=True))

    def test_t1_default_last_token_logits(self):
        hidden, logits = self.model.forward_no_cache(
            input_ids=torch.tensor([1], device="cuda", dtype=torch.int32),
            positions=torch.tensor([0], device="cuda", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            max_seqlen=1,
        )
        torch.cuda.synchronize()
        self.assertEqual(
            (tuple(hidden.shape), tuple(logits.shape)), ((1, 2048), (1, 248320))
        )
        self.assertTrue(torch.isfinite(hidden).all() and torch.isfinite(logits).all())

    def test_ragged_explicit_logits_indices(self):
        hidden, logits = self.model.forward_no_cache(
            input_ids=torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32),
            positions=torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32),
            max_seqlen=2,
            logits_indices=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        )
        torch.cuda.synchronize()
        self.assertEqual(
            (tuple(hidden.shape), tuple(logits.shape)), ((3, 2048), (2, 248320))
        )
        self.assertTrue(torch.isfinite(hidden).all() and torch.isfinite(logits).all())

    def test_real_chain_length_matrix_and_empty_batch(self):
        # Layers 0--2 take the FP16->NVFP4 routed MoE path and layer 3 takes
        # its GDN->full-attention transition; these are shape/finite checks,
        # not a statement about model quality.
        for length in (1, 4, 16, 17, 64, 65):
            with self.subTest(length=length):
                hidden, logits = self._forward([length], start_id=100 + length)
                self.assertEqual(tuple(hidden.shape), (length, 2048))
                self.assertEqual(tuple(logits.shape), (1, 248320))
                self.assertTrue(
                    torch.isfinite(hidden).all() and torch.isfinite(logits).all()
                )
        hidden, logits = self._forward([1, 4, 17, 65], start_id=300)
        self.assertEqual(
            (tuple(hidden.shape), tuple(logits.shape)), ((87, 2048), (4, 248320))
        )
        self.assertTrue(torch.isfinite(hidden).all() and torch.isfinite(logits).all())
        ids, positions, cu = self._packed([0])
        hidden, logits = self.model.forward_no_cache(
            input_ids=ids,
            positions=positions,
            cu_seqlens=cu,
            max_seqlen=0,
        )
        self.assertEqual(
            (tuple(hidden.shape), tuple(logits.shape)), ((0, 2048), (0, 248320))
        )

    def test_default_empty_logits_and_explicit_selection(self):
        hidden, default_logits = self._forward([0, 1, 0, 2], start_id=500)
        self.assertEqual(
            (tuple(hidden.shape), tuple(default_logits.shape)), ((3, 2048), (4, 248320))
        )
        torch.testing.assert_close(
            default_logits[0], torch.zeros_like(default_logits[0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            default_logits[2], torch.zeros_like(default_logits[2]), rtol=0, atol=0
        )
        explicit = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
        _, explicit_logits = self._forward(
            [0, 1, 0, 2], logits_indices=explicit, start_id=500
        )
        torch.testing.assert_close(
            explicit_logits, default_logits[[1, 3]], rtol=0, atol=0
        )

    def test_input_or_hidden_exclusive_and_packed_sequence_equivalence(self):
        ids, positions, cu = self._packed([17, 65], start_id=800)
        embedded = self.model.runner.embed(ids)
        with self.assertRaises(ValueError):
            self.model.forward_no_cache(
                input_ids=ids,
                hidden_states=embedded,
                positions=positions,
                cu_seqlens=cu,
                max_seqlen=65,
            )
        packed_hidden, packed_logits = self.model.forward_no_cache(
            hidden_states=embedded,
            positions=positions,
            cu_seqlens=cu,
            max_seqlen=65,
        )
        separate_hidden, separate_logits = [], []
        cursor = 0
        for length in (17, 65):
            end = cursor + length
            local_hidden, local_logits = self.model.forward_no_cache(
                hidden_states=embedded[cursor:end].contiguous(),
                positions=torch.arange(length, device="cuda", dtype=torch.int32),
                cu_seqlens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
                max_seqlen=length,
            )
            separate_hidden.append(local_hidden)
            separate_logits.append(local_logits)
            cursor = end
        hidden_nrmse = self._nrmse(packed_hidden, torch.cat(separate_hidden))
        logits_nrmse = self._nrmse(packed_logits, torch.cat(separate_logits))
        self._report(
            case="packed_vs_per_sequence",
            hidden_nrmse=hidden_nrmse,
            logits_nrmse=logits_nrmse,
            shape=list(packed_hidden.shape),
        )
        self.assertLessEqual(hidden_nrmse, 5e-3)
        self.assertLessEqual(logits_nrmse, 5e-3)

    def test_repeat_after_distinct_call_and_independent_global_oracles(self):
        hidden, logits = self._forward([65], start_id=1000)
        self._forward([4], start_id=1100)
        repeat_hidden, repeat_logits = self._forward([65], start_id=1000)
        torch.testing.assert_close(repeat_hidden, hidden, rtol=0, atol=0)
        torch.testing.assert_close(repeat_logits, logits, rtol=0, atol=0)

        # Independently inspect embedding, final RMS norm, and one <=4096
        # vocabulary block of the head.  This is a local numeric diagnostic,
        # not a whole-model quality oracle.
        runner = self.model.runner
        ids, positions, cu = self._packed([1], start_id=1200)
        embedded = runner.embed(ids)
        torch.testing.assert_close(
            embedded,
            runner.global_weights["embed_tokens"].data[ids.long()],
            rtol=0,
            atol=0,
        )
        before_final = runner.forward_hidden(
            embedded, positions=positions, cu_seqlens=cu, max_seqlen=1
        )
        expected_final = (
            before_final.float()
            * torch.rsqrt(before_final.float().square().mean(-1, keepdim=True) + 1e-6)
            * (1 + runner.global_weights["final_norm"].float())
        ).half()
        actual_final = runner.final_hidden(before_final)
        torch.testing.assert_close(actual_final, expected_final, rtol=2e-3, atol=2e-3)
        actual_logits = runner.logits(actual_final)
        head_block = runner.global_weights["lm_head"].data[:4096]
        expected_block = (actual_final.float() @ head_block.float().t()).half()
        torch.testing.assert_close(
            actual_logits[:, :4096], expected_block, rtol=2e-3, atol=2e-3
        )

    def test_short_sequences_ignore_an_inflated_batch_maximum(self):
        ids, positions, cu = self._packed([17], start_id=1300)
        reference_hidden, reference_logits = self.model.forward_no_cache(
            input_ids=ids,
            positions=positions,
            cu_seqlens=cu,
            max_seqlen=17,
        )
        for declared_max in (65, 2048):
            with self.subTest(declared_max=declared_max):
                hidden, logits = self.model.forward_no_cache(
                    input_ids=ids,
                    positions=positions,
                    cu_seqlens=cu,
                    max_seqlen=declared_max,
                )
                torch.testing.assert_close(hidden, reference_hidden, rtol=0, atol=0)
                torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)

    def test_int64_empty_and_63_64_65_sequence_isolation(self):
        lengths = [0, 63, 64, 65]
        ids, positions, cu = self._packed(lengths, start_id=1350)
        cu = cu.to(torch.int64)
        selected = torch.tensor([62, 126, 191], device="cuda", dtype=torch.int64)
        packed_hidden, packed_logits = self.model.forward_no_cache(
            input_ids=ids,
            positions=positions,
            cu_seqlens=cu,
            max_seqlen=65,
            logits_indices=selected,
        )
        separate_hidden, separate_logits = [], []
        cursor = 0
        for length in (63, 64, 65):
            end = cursor + length
            local_hidden, local_logits = self.model.forward_no_cache(
                input_ids=ids[cursor:end].contiguous(),
                positions=torch.arange(length, device="cuda", dtype=torch.int64),
                cu_seqlens=torch.tensor([0, length], device="cuda", dtype=torch.int64),
                max_seqlen=length,
            )
            separate_hidden.append(local_hidden)
            separate_logits.append(local_logits)
            cursor = end
        torch.testing.assert_close(
            packed_hidden, torch.cat(separate_hidden), rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed_logits, torch.cat(separate_logits), rtol=0, atol=0
        )

    def test_single_sequence_t2048_peak(self):
        self._forward(
            [1],
            logits_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
            start_id=1450,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        hidden, logits = self._forward([2048], start_id=1600)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        self._report(
            case="single_sequence_t2048",
            peak_bytes=peak,
            shape=list(hidden.shape),
            finite=bool(torch.isfinite(hidden).all() and torch.isfinite(logits).all()),
        )
        self.assertEqual(
            (tuple(hidden.shape), tuple(logits.shape)), ((2048, 2048), (1, 248320))
        )
        self.assertTrue(torch.isfinite(hidden).all() and torch.isfinite(logits).all())
        self.assertLessEqual(peak, 12 * 1024**3)

    def test_worst_ragged_t2048_peak(self):
        # Keep logits to one row; the test measures layer/GDN workspace rather
        # than a deliberately large default [B,vocab] result.
        lengths = [1] * 1983 + [65]
        indices = torch.tensor([2047], device="cuda", dtype=torch.int32)
        self._forward(
            [1],
            logits_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
            start_id=1400,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        hidden, logits = self._forward(lengths, logits_indices=indices, start_id=1500)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        self._report(
            case="worst_ragged_t2048",
            peak_bytes=peak,
            shape=list(hidden.shape),
            finite=bool(torch.isfinite(hidden).all() and torch.isfinite(logits).all()),
        )
        self.assertEqual(
            (tuple(hidden.shape), tuple(logits.shape)), ((2048, 2048), (1, 248320))
        )
        self.assertTrue(torch.isfinite(hidden).all() and torch.isfinite(logits).all())
        self.assertLessEqual(peak, 12 * 1024**3)


class TestZZLayer39FullAttention(V100TestCase):
    """Standalone final full-attention path with its FP16 routed experts."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not MODEL_DIR.is_dir():
            raise unittest.SkipTest("real checkpoint unavailable")
        cls.runner = Qwen35StatelessRunner(
            MODEL_DIR, layer_ids=[39], load_globals=False
        )

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        torch.cuda.empty_cache()
        super().tearDownClass()

    def test_layer39_full_attention_fp16_expert_route(self):
        hidden = torch.randn((3, 2048), device="cuda", dtype=torch.float16)
        result = self.runner.forward_layer(
            hidden,
            39,
            positions=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
            max_seqlen=3,
        )
        torch.cuda.synchronize()
        self.assertEqual(tuple(result.shape), (3, 2048))
        self.assertTrue(torch.isfinite(result).all())
