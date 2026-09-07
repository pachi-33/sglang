"""Real-weight checks for the single-request runner cache path.

These deliberately use one original layer at a time: loading a whole model is
not required to prove that a layer's prefill state is mathematically continuous
with its forced decode.  ``forward_hidden`` is the runner's no-cache oracle for
the same complete input prefix.
"""

import os
import unittest
from pathlib import Path
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase

import torch

from sglang.srt.layers.qwen3_5.runner import (
    FullAttentionLayerCache,
    GDNLayerCache,
    Qwen35StatelessRunner,
)

MODEL_DIR = Path(
    os.environ.get(
        "QWEN35_MODEL_DIR",
        "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
    )
)


class _StatefulSingleLayer:
    """Shared forced-decoding contract run once for a GDN and a Full layer."""

    layer_id: int
    cache_type: type[GDNLayerCache] | type[FullAttentionLayerCache]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not MODEL_DIR.is_dir():
            raise unittest.SkipTest("real Qwen3.5 checkpoint unavailable")
        cls.runner = Qwen35StatelessRunner(
            MODEL_DIR, layer_ids=[cls.layer_id], load_globals=False
        )

    @classmethod
    def tearDownClass(cls):
        del cls.runner
        torch.cuda.empty_cache()
        super().tearDownClass()

    @staticmethod
    def _nrmse(actual, expected):
        return (
            (actual.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-8)
        ).item()

    def _forward_no_cache(self, hidden):
        tokens = hidden.shape[0]
        return self.runner.forward_no_cache(
            hidden,
            positions=torch.arange(tokens, device="cuda", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
            max_seqlen=tokens,
        )

    def _assert_cache_layout(self, cache, capacity):
        self.assertEqual(cache.capacity, capacity)
        self.assertEqual(cache.consumed_len, 0)
        self.assertFalse(cache.poisoned)
        layer_cache = cache.layers[self.layer_id]
        self.assertIsInstance(layer_cache, self.cache_type)
        if isinstance(layer_cache, GDNLayerCache):
            self.assertEqual(tuple(layer_cache.conv_history.shape), (3, 8192))
            self.assertEqual(layer_cache.conv_history.dtype, torch.float16)
            self.assertEqual(tuple(layer_cache.recurrent_state.shape), (32, 128, 128))
            self.assertEqual(layer_cache.recurrent_state.dtype, torch.float32)
            # One GDN layer's share of the documented 15-layer cache budget.
            self.assertEqual(layer_cache.conv_history.nbytes, 48 * 1024)
            self.assertEqual(layer_cache.recurrent_state.nbytes, 2 * 1024**2)
        else:
            self.assertEqual(tuple(layer_cache.key.shape), (capacity, 2, 256))
            self.assertEqual(tuple(layer_cache.value.shape), (capacity, 2, 256))
            self.assertEqual(layer_cache.key.dtype, torch.float16)
            self.assertEqual(layer_cache.value.dtype, torch.float16)
            # K and V are independently 2 MiB at the production capacity.
            self.assertEqual(layer_cache.key.nbytes, capacity * 2 * 256 * 2)
            self.assertEqual(layer_cache.value.nbytes, capacity * 2 * 256 * 2)

    def _poison_during_execution(self, hidden):
        """Cause an internal cache write failure, not a caller validation error."""
        cache = self.runner.allocate_request_cache(capacity=4)
        layer_cache = cache.layers[self.layer_id]
        if isinstance(layer_cache, GDNLayerCache):
            self.runner.prefill_hidden(hidden[:3].contiguous(), cache=cache)
            # The decode kernel validates this cache tensor inside the guarded
            # execution section, so the cache must become poisoned.
            layer_cache.conv_history = layer_cache.conv_history.float()
            with self.assertRaises(ValueError):
                self.runner.decode_hidden(
                    hidden[3:].contiguous(), cache=cache, expected_prefix_len=3
                )
        else:
            # A short KV cache fails while prefill copies its third K row.
            layer_cache.key = layer_cache.key[:2]
            with self.assertRaises(RuntimeError):
                self.runner.prefill_hidden(hidden[:3].contiguous(), cache=cache)
        self.assertTrue(cache.poisoned)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            self.runner.decode_hidden(
                hidden[3:].contiguous(), cache=cache, expected_prefix_len=3
            )
        self.runner.reset_request_cache(cache)
        self.assertFalse(cache.poisoned)
        self.assertEqual(cache.consumed_len, 0)

    def test_prefill_forced_decode_cache_lifecycle(self):
        torch.manual_seed(3200 + self.layer_id)
        # Capacity four makes the successful forced decode immediately exercise
        # the subsequent capacity error without a long real-weight run.
        hidden = torch.randn((4, 2048), device="cuda", dtype=torch.float16)
        cache = self.runner.allocate_request_cache(capacity=4)
        self._assert_cache_layout(cache, 4)

        prefetched = self.runner.prefill_hidden(hidden[:3].contiguous(), cache=cache)
        reference = self._forward_no_cache(hidden)
        decoded = self.runner.decode_hidden(
            hidden[3:].contiguous(), cache=cache, expected_prefix_len=3
        )
        self.assertEqual(cache.consumed_len, 4)
        self.assertFalse(cache.poisoned)
        self.assertTrue(
            torch.isfinite(prefetched).all() and torch.isfinite(decoded).all()
        )
        self.assertLessEqual(self._nrmse(prefetched[-1:], reference[2:3]), 5e-3)
        self.assertLessEqual(self._nrmse(decoded, reference[3:]), 5e-3)

        # Contract errors before the guarded execution must not mutate or poison.
        with self.assertRaisesRegex(RuntimeError, "expected prefix"):
            self.runner.decode_hidden(
                hidden[3:].contiguous(), cache=cache, expected_prefix_len=3
            )
        self.assertEqual(cache.consumed_len, 4)
        self.assertFalse(cache.poisoned)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            self.runner.decode_hidden(
                hidden[3:].contiguous(), cache=cache, expected_prefix_len=4
            )
        self.assertEqual(cache.consumed_len, 4)

        self.runner.reset_request_cache(cache)
        self.assertEqual(cache.consumed_len, 0)
        self.assertFalse(cache.poisoned)
        layer_cache = cache.layers[self.layer_id]
        if isinstance(layer_cache, GDNLayerCache):
            torch.testing.assert_close(
                layer_cache.conv_history,
                torch.zeros_like(layer_cache.conv_history),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                layer_cache.recurrent_state,
                torch.zeros_like(layer_cache.recurrent_state),
                rtol=0,
                atol=0,
            )

        self._poison_during_execution(hidden)


class TestStatefulLayer0GDN(_StatefulSingleLayer, V100TestCase):
    layer_id = 0
    cache_type = GDNLayerCache


class TestStatefulLayer3FullAttention(_StatefulSingleLayer, V100TestCase):
    layer_id = 3
    cache_type = FullAttentionLayerCache


class TestStatefulLayer20GDN(_StatefulSingleLayer, V100TestCase):
    """Exercise the 4070-side GDN slice beyond the recurrent short path."""

    layer_id = 20
    cache_type = GDNLayerCache

    def test_chunk_prefill_then_decode_separate_cuda_inputs(self):
        """A 65-token WY prefill must continue with one recurrent decode.

        Keep the prefix and decode rows in separate live CUDA allocations.  The
        pipeline receives those rows independently over its CPU boundary, so a
        view into one contiguous 66-row allocation is not representative here.
        ``forward_no_cache`` remains the complete-prefix oracle.
        """
        torch.manual_seed(3220)
        prefix = torch.randn((65, 2048), device="cpu", dtype=torch.float16).to("cuda")
        decode_input = torch.randn((1, 2048), device="cpu", dtype=torch.float16).to(
            "cuda"
        )
        self.assertNotEqual(
            prefix.untyped_storage().data_ptr(),
            decode_input.untyped_storage().data_ptr(),
        )
        full_prefix = torch.empty((66, 2048), device="cuda", dtype=torch.float16)
        full_prefix[:65].copy_(prefix)
        full_prefix[65:].copy_(decode_input)

        # Run the stateless oracle before cache mutation: this also explicitly
        # verifies the real layer's >64-token chunk path.
        reference = self._forward_no_cache(full_prefix)
        cache = self.runner.allocate_request_cache(capacity=66)
        self._assert_cache_layout(cache, 66)
        prefetched = self.runner.prefill_hidden(prefix, cache=cache)
        self.assertEqual(cache.consumed_len, 65)
        self.assertFalse(cache.poisoned)
        layer_cache = cache.layers[self.layer_id]
        self.assertIsInstance(layer_cache, GDNLayerCache)
        self.assertTrue(torch.isfinite(layer_cache.recurrent_state).all())
        self.assertGreater(layer_cache.recurrent_state.abs().max().item(), 0.0)

        decoded = self.runner.decode_hidden(
            decode_input, cache=cache, expected_prefix_len=65
        )
        self.assertEqual(cache.consumed_len, 66)
        self.assertFalse(cache.poisoned)
        self.assertTrue(
            torch.isfinite(prefetched).all() and torch.isfinite(decoded).all()
        )
        self.assertLessEqual(self._nrmse(prefetched[-1:], reference[64:65]), 5e-3)
        self.assertLessEqual(self._nrmse(decoded, reference[65:]), 5e-3)

        self.runner.reset_request_cache(cache)
        self.assertEqual(cache.consumed_len, 0)
        self.assertFalse(cache.poisoned)
        torch.testing.assert_close(
            layer_cache.conv_history,
            torch.zeros_like(layer_cache.conv_history),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer_cache.recurrent_state,
            torch.zeros_like(layer_cache.recurrent_state),
            rtol=0,
            atol=0,
        )


class TestStatefulLayer23FullAttention(_StatefulSingleLayer, V100TestCase):
    layer_id = 23
    cache_type = FullAttentionLayerCache
