import unittest
from test.Qwen3_5MoECompat.reference.gdn import (
    recurrent,
    recurrent_decode,
    recurrent_vectorized,
)
from test.Qwen3_5MoECompat.unit.test_environment import V100TestCase

import torch

from sglang.srt.layers.qwen3_5.gdn import (
    chunk_gdn,
    depthwise_conv4_silu,
    depthwise_conv4_silu_decode,
    l2_normalize_qk,
    prepare_gates,
    recurrent_gdn,
    recurrent_gdn_decode,
    recurrent_gdn_short_output,
)
from sglang.srt.layers.qwen3_5.kernels.gdn_chunk import (
    compute_gram_a16,
    compute_r_state16,
    compute_wy16,
    stream_gdn16,
)
from sglang.srt.layers.qwen3_5.runner import _validate_metadata


class TestGDNRecurrent(V100TestCase):
    def _inputs(self, lengths, seed):
        torch.manual_seed(seed)
        tokens = sum(lengths)
        q = torch.randn((tokens, 16, 128), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn((tokens, 32, 128), device="cuda", dtype=torch.float16)
        a = torch.randn((tokens, 32), device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        alog = torch.randn((32,), device="cuda", dtype=torch.float16)
        dt = torch.randn_like(alog)
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        q, k = l2_normalize_qk(q, k)
        decay, beta = prepare_gates(a, b, alog, dt)
        return q, k, v, decay, beta, cu

    def _check_recurrent(self, lengths, seed):
        q, k, v, decay, beta, cu = self._inputs(lengths, seed)
        actual, state = recurrent_gdn(q, k, v, decay, beta, cu, max(lengths))
        expected, expected_state = recurrent(q, k, v, decay, beta, cu.cpu())
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-4)
        repeat, repeat_state = recurrent_gdn(q, k, v, decay, beta, cu, max(lengths))
        torch.testing.assert_close(repeat, actual, rtol=0, atol=0)
        torch.testing.assert_close(repeat_state, state, rtol=0, atol=0)

    def test_recurrent_single_lengths(self):
        for seed, length in enumerate((1, 17, 65), start=61):
            with self.subTest(length=length):
                self._check_recurrent([length], seed)

    def test_recurrent_ragged_sequence_isolation(self):
        self._check_recurrent([1, 17, 47], 72)

    def test_conv4_silu_ragged_boundaries(self):
        torch.manual_seed(80)
        lengths = [1, 3, 5]
        tokens, channels = sum(lengths), 7
        x = torch.randn((tokens, channels), device="cuda", dtype=torch.float16)
        weight = torch.randn((channels, 4), device="cuda", dtype=torch.float16)
        bias = torch.randn((channels,), device="cuda", dtype=torch.float16)
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        actual = depthwise_conv4_silu(x, weight, bias, cu)
        expected = torch.empty_like(x)
        for start, end in zip(cu[:-1].cpu().tolist(), cu[1:].cpu().tolist()):
            for t in range(start, end):
                total = bias.float().clone()
                for tap in range(4):
                    index = t - 3 + tap
                    if index >= start:
                        total += x[index].float() * weight[:, tap].float()
                expected[t] = torch.nn.functional.silu(total).half()
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-3)
        actual_no_bias = depthwise_conv4_silu(x, weight, None, cu)
        expected_no_bias = torch.empty_like(x)
        for start, end in zip(cu[:-1].cpu().tolist(), cu[1:].cpu().tolist()):
            for t in range(start, end):
                total = torch.zeros((channels,), device="cuda", dtype=torch.float32)
                for tap in range(4):
                    index = t - 3 + tap
                    if index >= start:
                        total += x[index].float() * weight[:, tap].float()
                expected_no_bias[t] = torch.nn.functional.silu(total).half()
        torch.testing.assert_close(
            actual_no_bias, expected_no_bias, rtol=3e-3, atol=5e-3
        )

    def test_conv4_silu_checkpoint_channel_width(self):
        torch.manual_seed(81)
        channels = 8192
        x = torch.randn((4, channels), device="cuda", dtype=torch.float16)
        weight = torch.randn((channels, 4), device="cuda", dtype=torch.float16)
        cu = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
        actual = depthwise_conv4_silu(x, weight, None, cu)
        expected = torch.empty_like(x)
        for t in range(4):
            total = torch.zeros((channels,), device="cuda", dtype=torch.float32)
            for tap in range(4):
                index = t - 3 + tap
                if index >= 0:
                    total += x[index].float() * weight[:, tap].float()
            expected[t] = torch.nn.functional.silu(total).half()
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-3)

    def test_conv4_silu_decode_history_for_short_prompts_and_splits(self):
        """The cache tail must be raw inputs, zero-left-padded, oldest first."""
        torch.manual_seed(811)
        channels = 257
        weight = torch.randn((channels, 4), device="cuda", dtype=torch.float16)
        bias = torch.randn((channels,), device="cuda", dtype=torch.float16)
        for tokens in (1, 2, 3, 4):
            with self.subTest(tokens=tokens, mode="all_decode"):
                x = torch.randn((tokens, channels), device="cuda", dtype=torch.float16)
                cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
                expected = depthwise_conv4_silu(x, weight, bias, cu)
                history = torch.zeros((3, channels), device="cuda", dtype=torch.float16)
                actual = torch.cat(
                    [
                        depthwise_conv4_silu_decode(x[t : t + 1], weight, bias, history)
                        for t in range(tokens)
                    ]
                )
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-3)
                tail = torch.zeros_like(history)
                tail[-min(tokens, 3) :] = x[-3:]
                torch.testing.assert_close(history, tail, rtol=0, atol=0)

        # Decode receives a one-row slice of the merged [QKV,Z] projection.
        # PyTorch considers this view contiguous even though its singleton row
        # stride exceeds its width.  The Conv producer must canonicalize its
        # output rather than propagating that misleading stride downstream.
        merged = torch.randn((1, channels + 31), device="cuda", dtype=torch.float16)
        row_view = merged[:, :channels]
        self.assertEqual(row_view.stride(), (channels + 31, 1))
        history = torch.zeros((3, channels), device="cuda", dtype=torch.float16)
        canonical = depthwise_conv4_silu_decode(row_view, weight, bias, history)
        self.assertEqual(canonical.stride(), (channels, 1))
        self.assertEqual(canonical.untyped_storage().nbytes(), channels * 2)

        # Simulate prefill saving its tail, then continue at each short split.
        x = torch.randn((9, channels), device="cuda", dtype=torch.float16)
        expected = depthwise_conv4_silu(
            x, weight, bias, torch.tensor([0, 9], device="cuda", dtype=torch.int32)
        )
        for split in (1, 2, 3, 4):
            with self.subTest(split=split, mode="prefill_then_decode"):
                history = torch.zeros((3, channels), device="cuda", dtype=torch.float16)
                kept = min(split, 3)
                history[-kept:] = x[split - kept : split]
                tail = torch.cat(
                    [
                        depthwise_conv4_silu_decode(x[t : t + 1], weight, bias, history)
                        for t in range(split, x.shape[0])
                    ]
                )
                torch.testing.assert_close(tail, expected[split:], rtol=3e-3, atol=5e-3)

    def test_recurrent_gdn_decode_nonzero_state_inplace_and_out(self):
        q, k, v, decay, beta, _ = self._inputs([3], 812)
        state = torch.randn((32, 128, 128), device="cuda", dtype=torch.float32)
        expected_out, expected_state = recurrent_decode(
            q[:1], k[:1], v[:1], decay[:1], beta[:1], state
        )
        original_ptr = state.data_ptr()
        out = torch.empty((1, 32, 128), device="cuda", dtype=torch.float32)
        actual = recurrent_gdn_decode(
            q[:1], k[:1], v[:1], decay[:1], beta[:1], state, out=out
        )
        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertEqual(state.data_ptr(), original_ptr)
        torch.testing.assert_close(actual, expected_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-4)

        # The same cache is continuously updated rather than recreated.
        expected_out, expected_state = recurrent_decode(
            q[1:2], k[1:2], v[1:2], decay[1:2], beta[1:2], expected_state
        )
        actual = recurrent_gdn_decode(
            q[1:2], k[1:2], v[1:2], decay[1:2], beta[1:2], state
        )
        torch.testing.assert_close(actual, expected_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-4)

        overlapping = state.flatten()[128 : 128 + 32 * 128].view(1, 32, 128)
        with self.assertRaisesRegex(ValueError, "must not alias state"):
            recurrent_gdn_decode(
                q[2:3],
                k[2:3],
                v[2:3],
                decay[2:3],
                beta[2:3],
                state,
                out=overlapping,
            )

    def test_chunk_final_state_continues_through_recurrent_decode(self):
        """The WY chunk state has the exact [head,V,K] cache layout decoder uses."""
        q, k, v, decay, beta, _ = self._inputs([68], 813)
        prefix_cu = torch.tensor([0, 65], device="cuda", dtype=torch.int32)
        _, state = chunk_gdn(
            q[:65], k[:65], v[:65], decay[:65], beta[:65], prefix_cu, 65
        )
        state = state[0]
        decoded = []
        for token in range(65, 68):
            decoded.append(
                recurrent_gdn_decode(
                    q[token : token + 1],
                    k[token : token + 1],
                    v[token : token + 1],
                    decay[token : token + 1],
                    beta[token : token + 1],
                    state,
                )
            )
        expected, expected_state = recurrent_vectorized(
            q, k, v, decay, beta, torch.tensor([0, 68], device="cpu", dtype=torch.int32)
        )
        actual = torch.cat(decoded)
        for value, reference in ((actual, expected[65:]), (state, expected_state[0])):
            nrmse = (
                value - reference
            ).square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-8)
            self.assertLessEqual(nrmse.item(), 5e-3)

    def _check_chunk_nrmse(self, lengths, seed, slow_decay=False):
        q, k, v, decay, beta, cu = self._inputs(lengths, seed)
        if slow_decay:
            decay.fill_(-1e-4)
        actual, state = chunk_gdn(q, k, v, decay, beta, cu, max(lengths))
        expected, expected_state = recurrent_vectorized(q, k, v, decay, beta, cu.cpu())
        nrmse = (
            actual - expected
        ).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
        state_nrmse = (
            state - expected_state
        ).square().mean().sqrt() / expected_state.square().mean().sqrt().clamp_min(1e-8)
        self.assertLessEqual(nrmse.item(), 5e-3)
        self.assertLessEqual(state_nrmse.item(), 5e-3)
        return (q, k, v, decay, beta, cu), actual, state

    def test_chunk_bt16_long_independent_recurrence(self):
        self._check_chunk_nrmse([2048], 91, slow_decay=True)

    def test_chunk_bt16_streamed_skewed_ragged(self):
        # This has 128 sequences but only one carries most chunks.  It catches
        # accidental [B, ceil(max_seqlen / 16), ...] workspace allocation.
        self._check_chunk_nrmse([1] * 127 + [1921], 901, slow_decay=True)

    def test_chunk_bt16_ragged_tails_and_repeatability(self):
        inputs, actual, state = self._check_chunk_nrmse([1, 17, 65], 92)
        repeat, repeat_state = chunk_gdn(*inputs, max_seqlen=65)
        torch.testing.assert_close(repeat, actual, rtol=0, atol=0)
        torch.testing.assert_close(repeat_state, state, rtol=0, atol=0)

    def test_short_output_overwrite_boundaries_and_int64_empty(self):
        # The helper is used only after WY when a batch also contains a long
        # sequence.  It must overwrite 63/64-token documents, leave 65 alone,
        # and do nothing for an empty int64 segment.
        lengths = [0, 63, 64, 65]
        q, k, v, decay, beta, _ = self._inputs(lengths, 904)
        cu = torch.tensor([0, 0, 63, 127, 192], device="cuda", dtype=torch.int64)
        wy, _ = chunk_gdn(q, k, v, decay, beta, cu, 65)
        before = wy.clone()
        exact, _ = recurrent_gdn(q, k, v, decay, beta, cu, 65)
        recurrent_gdn_short_output(q, k, v, decay, beta, cu, 65, wy)
        torch.testing.assert_close(wy[:127], exact[:127], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(wy[127:], before[127:], rtol=0, atol=0)

    def test_chunk_bt16_exact_block_multiples(self):
        for seed, length in enumerate((16, 128, 512), start=97):
            with self.subTest(length=length):
                self._check_chunk_nrmse([length], seed)

    def test_prepare_gates_stable_softplus_and_l2(self):
        values = torch.tensor(
            [-30.0, -20.0, 0.0, 20.0, 100.0], device="cuda", dtype=torch.float16
        ).view(-1, 1)
        zeros = torch.zeros_like(values)
        alog = torch.zeros((1,), device="cuda", dtype=torch.float16)
        decay, beta = prepare_gates(values, zeros, alog, zeros[0])
        expected = -torch.nn.functional.softplus(values.float())
        torch.testing.assert_close(decay, expected, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(decay[:2], expected[:2], rtol=2e-3, atol=0)
        torch.testing.assert_close(beta, torch.full_like(beta, 0.5), rtol=0, atol=0)
        q = torch.randn((3, 16, 128), device="cuda", dtype=torch.float16)
        nq, nk = l2_normalize_qk(q, q.clone())
        torch.testing.assert_close(
            nq.float().square().sum(-1),
            torch.ones((3, 16), device="cuda"),
            rtol=3e-3,
            atol=3e-3,
        )
        torch.testing.assert_close(nk, nq, rtol=0, atol=0)

    def test_int64_cu_and_empty_segments(self):
        q, k, v, decay, beta, _ = self._inputs([1, 17], 93)
        cu = torch.tensor([0, 1, 1, 18, 18], device="cuda", dtype=torch.int64)
        actual, state = chunk_gdn(q, k, v, decay, beta, cu, 17)
        expected, expected_state = recurrent_vectorized(q, k, v, decay, beta, cu.cpu())
        torch.testing.assert_close(actual, expected, rtol=6e-3, atol=6e-3)
        torch.testing.assert_close(state, expected_state, rtol=6e-3, atol=6e-3)
        torch.testing.assert_close(state[1], torch.zeros_like(state[1]), rtol=0, atol=0)
        torch.testing.assert_close(state[3], torch.zeros_like(state[3]), rtol=0, atol=0)

    def test_recurrent_relative_l2_and_state_precision(self):
        q, k, v, decay, beta, cu = self._inputs([1, 17, 65], 902)
        actual, state = recurrent_gdn(q, k, v, decay, beta, cu, 65)
        expected, expected_state = recurrent_vectorized(q, k, v, decay, beta, cu.cpu())
        output_rel = torch.linalg.vector_norm(
            (actual - expected).flatten(1), dim=1
        ) / torch.linalg.vector_norm(expected.flatten(1), dim=1).clamp_min(1e-8)
        state_rel = torch.linalg.vector_norm(
            (state - expected_state).flatten(1), dim=1
        ) / torch.linalg.vector_norm(expected_state.flatten(1), dim=1).clamp_min(1e-8)
        self.assertTrue(torch.isfinite(output_rel).all())
        self.assertTrue(torch.isfinite(state_rel).all())
        self.assertLessEqual(
            output_rel.max().item(),
            1e-4,
            f"output p99={torch.quantile(output_rel, .99).item():.3e}",
        )
        self.assertLessEqual(
            state_rel.max().item(),
            1e-4,
            f"state p99={torch.quantile(state_rel, .99).item():.3e}",
        )

    def test_gdn_and_runner_metadata_reject_invalid_max_or_cpu_metadata(self):
        q, k, v, decay, beta, cu = self._inputs([1], 903)
        with self.assertRaises(ValueError):
            chunk_gdn(q, k, v, decay, beta, cu, 0)
        with self.assertRaises(ValueError):
            chunk_gdn(q, k, v, decay, beta, cu, -1)
        with self.assertRaises(ValueError):
            _validate_metadata(1, torch.zeros((1,), dtype=torch.int32), cu, 1, q.device)
        with self.assertRaises(ValueError):
            _validate_metadata(
                1, torch.zeros((1,), device="cuda", dtype=torch.int32), cu, 0, q.device
            )

    def test_separated_bt16_gram_and_inverse_four_chunks(self):
        lengths = [49, 17]
        q, k, v, decay, beta, cu = self._inputs(lengths, 94)
        gram, actual_a = compute_gram_a16(q, k, decay, beta, cu, max(lengths))
        for seq, (start, end) in enumerate(
            zip(cu[:-1].cpu().tolist(), cu[1:].cpu().tolist())
        ):
            for chunk, block_start in enumerate(range(start, end, 16)):
                n = min(16, end - block_start)
                kh = (
                    k[block_start : block_start + n]
                    .repeat_interleave(2, dim=1)
                    .permute(1, 0, 2)
                    .float()
                )
                expected_gram = torch.bmm(kh, kh.transpose(1, 2))
                torch.testing.assert_close(
                    gram[seq, chunk, :, :n, :n], expected_gram, rtol=3e-3, atol=3e-3
                )
                g = torch.cumsum(
                    decay[block_start : block_start + n].transpose(0, 1), dim=1
                )
                l = (
                    beta[block_start : block_start + n]
                    .transpose(0, 1)
                    .float()
                    .unsqueeze(2)
                    * expected_gram
                    * torch.exp(g[:, :, None] - g[:, None, :])
                )
                l = torch.tril(l, diagonal=-1)
                eye = torch.eye(n, device="cuda", dtype=torch.float32).expand(
                    32, -1, -1
                )
                expected_a = torch.linalg.solve_triangular(
                    eye + l, eye, upper=False
                ).half()
                torch.testing.assert_close(
                    actual_a[seq, chunk, :, :n, :n], expected_a, rtol=3e-3, atol=3e-3
                )

    def test_separated_bt16_wy_real_dim_ragged(self):
        lengths = [49, 17]
        q, k, v, decay, beta, cu = self._inputs(lengths, 95)
        _, a16 = compute_gram_a16(q, k, decay, beta, cu, max(lengths))
        actual_u, actual_w = compute_wy16(q, k, v, decay, beta, cu, max(lengths), a16)
        expected_u, expected_w = [], []
        actual_u_active, actual_w_active = [], []
        for seq, (start, end) in enumerate(
            zip(cu[:-1].cpu().tolist(), cu[1:].cpu().tolist())
        ):
            for chunk, block_start in enumerate(range(start, end, 16)):
                n = min(16, end - block_start)
                aa = a16[seq, chunk, :, :n, :n].float()
                vv = (
                    beta[block_start : block_start + n]
                    .transpose(0, 1)
                    .float()
                    .unsqueeze(2)
                    * v[block_start : block_start + n].permute(1, 0, 2).float()
                ).half()
                kh = (
                    k[block_start : block_start + n]
                    .repeat_interleave(2, dim=1)
                    .permute(1, 0, 2)
                    .float()
                )
                g = torch.cumsum(
                    decay[block_start : block_start + n].transpose(0, 1), dim=1
                )
                kk = (
                    beta[block_start : block_start + n]
                    .transpose(0, 1)
                    .float()
                    .unsqueeze(2)
                    * torch.exp(g).unsqueeze(2)
                    * kh
                ).half()
                expected_u.append(torch.bmm(aa, vv.float()).half())
                expected_w.append(torch.bmm(aa, kk.float()).half())
                actual_u_active.append(actual_u[seq, chunk, :, :n])
                actual_w_active.append(actual_w[seq, chunk, :, :n])
        for actual, expected in (
            (
                torch.cat([x.reshape(-1, 128) for x in actual_u_active]),
                torch.cat([x.reshape(-1, 128) for x in expected_u]),
            ),
            (
                torch.cat([x.reshape(-1, 128) for x in actual_w_active]),
                torch.cat([x.reshape(-1, 128) for x in expected_w]),
            ),
        ):
            self.assertTrue(torch.isfinite(actual).all())
            nrmse = (
                actual.float() - expected.float()
            ).square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(
                1e-8
            )
            self.assertLessEqual(nrmse.item(), 2e-3)

    def test_separated_bt16_residual_and_state_ragged(self):
        lengths = [49, 17]
        q, k, v, decay, beta, cu = self._inputs(lengths, 96)
        _, a16 = compute_gram_a16(q, k, decay, beta, cu, max(lengths))
        u16, w16 = compute_wy16(q, k, v, decay, beta, cu, max(lengths), a16)
        actual_h, actual_r32, actual_r16, actual_rd, actual_state = compute_r_state16(
            k, decay, cu, max(lengths), u16, w16
        )
        state = torch.zeros_like(actual_state)
        expected_h, expected_r, expected_rd = [], [], []
        actual_h_active, actual_r_active, actual_rd_active = [], [], []
        for seq, (start, end) in enumerate(
            zip(cu[:-1].cpu().tolist(), cu[1:].cpu().tolist())
        ):
            for chunk, block_start in enumerate(range(start, end, 16)):
                n = min(16, end - block_start)
                h16 = state[seq].half()
                r32 = u16[seq, chunk, :, :n].float() - torch.bmm(
                    w16[seq, chunk, :, :n].float(), h16.float().transpose(1, 2)
                )
                g = torch.cumsum(
                    decay[block_start : block_start + n].transpose(0, 1), dim=1
                )
                rd = (r32 * torch.exp(g[:, -1:] - g).unsqueeze(2)).half()
                kh = (
                    k[block_start : block_start + n]
                    .repeat_interleave(2, dim=1)
                    .permute(1, 0, 2)
                    .float()
                )
                state[seq] = torch.exp(g[:, -1]).view(32, 1, 1) * state[
                    seq
                ] + torch.bmm(rd.float().transpose(1, 2), kh)
                expected_h.append(h16)
                expected_r.append(r32)
                expected_rd.append(rd.transpose(1, 2))
                actual_h_active.append(actual_h[seq, chunk])
                actual_r_active.append(actual_r32[seq, chunk, :, :n])
                actual_rd_active.append(actual_rd[seq, chunk, :, :, :n])
        for actual, expected in (
            (
                torch.cat([x.reshape(-1) for x in actual_h_active]),
                torch.cat([x.reshape(-1) for x in expected_h]),
            ),
            (
                torch.cat([x.reshape(-1) for x in actual_r_active]),
                torch.cat([x.reshape(-1) for x in expected_r]),
            ),
            (
                torch.cat([x.reshape(-1) for x in actual_rd_active]),
                torch.cat([x.reshape(-1) for x in expected_rd]),
            ),
            (actual_state.reshape(-1), state.reshape(-1)),
        ):
            self.assertTrue(torch.isfinite(actual).all())
            nrmse = (
                actual.float() - expected.float()
            ).square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(
                1e-8
            )
            self.assertLessEqual(nrmse.item(), 2e-3)
        # Fresh allocations make repeated current-call evaluations bitwise stable.
        repeat = compute_r_state16(k, decay, cu, max(lengths), u16, w16)
        torch.testing.assert_close(repeat[-1], actual_state, rtol=0, atol=0)
        empty_cu = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
        empty_k = k[:0]
        empty_decay, empty_u, empty_w = decay[:0], u16[:1, :1], w16[:1, :1]
        empty = compute_r_state16(empty_k, empty_decay, empty_cu, 0, empty_u, empty_w)
        torch.testing.assert_close(
            empty[-1], torch.zeros_like(empty[-1]), rtol=0, atol=0
        )

    def test_merged_projection_row_views_match_contiguous_gdn_inputs(self):
        torch.manual_seed(916)
        tokens = 3
        merged = torch.randn((tokens, 12288), device="cuda", dtype=torch.float16)
        qkv = merged[:, :8192]
        weight = torch.randn((8192, 4), device="cuda", dtype=torch.float16)
        cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        torch.testing.assert_close(
            depthwise_conv4_silu(qkv, weight, None, cu),
            depthwise_conv4_silu(qkv.contiguous(), weight, None, cu),
            rtol=0,
            atol=0,
        )
        ba = torch.randn((tokens, 64), device="cuda", dtype=torch.float16)
        b, a = ba[:, :32], ba[:, 32:]
        alog = torch.randn((32,), device="cuda", dtype=torch.float16)
        dt = torch.randn_like(alog)
        decay, beta = prepare_gates(a, b, alog, dt)
        expected_decay, expected_beta = prepare_gates(
            a.contiguous(), b.contiguous(), alog, dt
        )
        torch.testing.assert_close(decay, expected_decay, rtol=0, atol=0)
        torch.testing.assert_close(beta, expected_beta, rtol=0, atol=0)

    def test_gdn_rejects_overlapping_or_nonunit_inner_stride_rows(self):
        bad = torch.empty((1, 16384), device="cuda", dtype=torch.float16)[:, ::2]
        weight = torch.empty((8192, 4), device="cuda", dtype=torch.float16)
        cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "unit-inner-stride"):
            depthwise_conv4_silu(bad, weight, None, cu)
        a = torch.empty((1, 64), device="cuda", dtype=torch.float16)[:, ::2]
        params = torch.empty((32,), device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "unit-inner-stride"):
            prepare_gates(a, a, params, params)

    def test_stream_reset_poisoned_state_and_empty_sequences(self):
        lengths = [0, 1, 17, 65, 0]
        q, k, v, decay, beta, cu = self._inputs(lengths, 990)
        expected, expected_state = chunk_gdn(q, k, v, decay, beta, cu, 65)
        poisoned = torch.full_like(expected_state, float("nan"))
        actual = torch.empty_like(expected)
        stream_gdn16(q, k, v, decay, beta, cu, 65, out=actual, state=poisoned)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(poisoned, expected_state, rtol=0, atol=0)
