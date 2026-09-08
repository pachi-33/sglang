"""CPU-only lifecycle tests for the Qwen3.5 single-GPU entry point."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.qwen3_5 import single_gpu


class _FakeRunner:
    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.embedded = []
        self.prefill_lengths = []
        self.decode_calls = []
        self.trace_steps = []
        self.reset_calls = 0
        self.closed = False
        self.failed = False
        self.fail_fatally = False
        self.prefill_error = None
        self.reset_error = None
        self.expert_stats = {"state": "READY", "cache_hits": 3}

    def allocate_request_cache(self, capacity):
        self.cache = SimpleNamespace(capacity=capacity, consumed_len=0, poisoned=False)
        return self.cache

    def embed(self, input_ids):
        values = input_ids.tolist()
        self.embedded.append(values)
        return input_ids.to(dtype=torch.float32).view(-1, 1)

    def prefill_hidden(self, hidden, *, cache, expert_trace_step=None):
        if self.prefill_error is not None:
            if self.fail_fatally:
                self.failed = True
                cache.poisoned = True
            raise self.prefill_error
        self.prefill_lengths.append(hidden.shape[0])
        if expert_trace_step is not None:
            self.trace_steps.append(("prefill", expert_trace_step))
        cache.consumed_len = hidden.shape[0]
        return hidden

    def decode_hidden(
        self, hidden, *, cache, expected_prefix_len, expert_trace_step=None
    ):
        self.decode_calls.append(
            (hidden.to(dtype=torch.int64).view(-1).tolist(), expected_prefix_len)
        )
        if expected_prefix_len != cache.consumed_len:
            raise AssertionError("bad expected prefix")
        if expert_trace_step is not None:
            self.trace_steps.append(("decode", expert_trace_step))
        cache.consumed_len += 1
        return hidden

    @staticmethod
    def final_hidden(hidden):
        return hidden

    def logits(self, hidden):
        if not self.outputs:
            raise AssertionError("fake runner has no scripted token")
        token = self.outputs.pop(0)
        logits = torch.full(
            (1, single_gpu.MODEL_VOCAB_SIZE), -10.0, dtype=torch.float32
        )
        logits[0, token] = 10.0
        # This invalid tokenizer-tail logit would win unless production masking
        # is applied before argmax.
        logits[0, single_gpu.TOKENIZER_VOCAB_SIZE] = 100.0
        return logits

    def reset_request_cache(self, cache):
        self.reset_calls += 1
        if self.reset_error is not None:
            raise self.reset_error
        cache.consumed_len = 0
        cache.poisoned = False

    def close(self):
        self.closed = True


class _FakeTraceSession:
    instances = []
    finalize_error = None

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.steps = []
        self.commits = []
        self.finalize_calls = []
        type(self).instances.append(self)

    def begin_step(self, **kwargs):
        step = SimpleNamespace(**kwargs)
        self.steps.append(step)
        return step

    def commit_step(self, step, sampled_token_id):
        self.commits.append((step, sampled_token_id))

    def finalize(self, **kwargs):
        self.finalize_calls.append(kwargs)
        if type(self).finalize_error is not None:
            raise type(self).finalize_error


class TestSingleGPU(unittest.TestCase):
    def setUp(self):
        _FakeTraceSession.instances.clear()
        _FakeTraceSession.finalize_error = None
        self.original_tensor = torch.tensor
        self.cuda_patches = (
            mock.patch.object(single_gpu.torch.cuda, "is_available", return_value=True),
            mock.patch.object(single_gpu.torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                single_gpu.torch.cuda,
                "get_device_capability",
                return_value=(7, 0),
            ),
            mock.patch.object(
                single_gpu.torch,
                "tensor",
                side_effect=lambda values, **kwargs: self.original_tensor(
                    values, dtype=kwargs.get("dtype")
                ),
            ),
            mock.patch.object(single_gpu, "_reset_peak_memory_stats"),
            mock.patch.object(
                single_gpu,
                "_cuda_memory_stats",
                return_value={
                    "allocated_bytes": 10,
                    "reserved_bytes": 20,
                    "peak_allocated_bytes": 30,
                    "peak_reserved_bytes": 40,
                    "total_memory_bytes": 100,
                    "free_after_bytes": 60,
                    "peak_reserved_margin_bytes": 60,
                },
            ),
        )
        for patcher in self.cuda_patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.cuda_patches):
            patcher.stop()

    def _backend(self, runner, **kwargs):
        with mock.patch.object(
            single_gpu, "Qwen35StatelessRunner", return_value=runner
        ) as constructor:
            backend = single_gpu.Qwen35SingleGPU(
                "/model",
                expert_pack_manifest="/pack/manifest.json",
                **kwargs,
            )
        return backend, constructor

    def test_constructs_complete_offloaded_runner_and_cache(self):
        runner = _FakeRunner()
        backend, constructor = self._backend(
            runner,
            capacity=123,
            expert_cache_mib=6144,
            expert_stage_slots=8,
            expert_io_workers=2,
            stats_path="/tmp/stats.json",
        )
        try:
            args, kwargs = constructor.call_args
            self.assertEqual(args[0], "/model")
            self.assertEqual(tuple(args[1]), tuple(range(40)))
            self.assertEqual(kwargs["device"], torch.device("cuda:0"))
            self.assertTrue(kwargs["load_globals"])
            config = kwargs["expert_offload"]
            self.assertEqual(config.cache_mib, 6144)
            self.assertEqual(config.stage_slots, 8)
            self.assertEqual(config.io_workers, 2)
            self.assertEqual(backend.cache.capacity, 123)
            self.assertEqual(backend.stats["cache_hits"], 3)
        finally:
            backend.close()

    def test_prefill_supplies_first_token_then_runs_r_minus_one_decodes(self):
        runner = _FakeRunner((31, 32, 33))
        backend, _ = self._backend(runner, capacity=8)
        try:
            generated = backend.generate_ids(
                [1, 2, 3], max_new_tokens=3, eos_token_ids=()
            )
        finally:
            backend.close()
        self.assertEqual(generated, [31, 32, 33])
        self.assertEqual(runner.prefill_lengths, [3])
        self.assertEqual(runner.embedded, [[1, 2, 3], [31], [32]])
        self.assertEqual(runner.decode_calls, [([31], 3), ([32], 4)])
        self.assertEqual(runner.reset_calls, 1)
        self.assertEqual(runner.cache.consumed_len, 0)
        stats = backend.stats
        self.assertEqual(stats["request_count"], 1)
        self.assertEqual(stats["last_generation"]["profile"], "cold")
        self.assertEqual(stats["last_generation"]["prompt_tokens"], 3)
        self.assertEqual(stats["last_generation"]["completion_tokens"], 3)
        self.assertEqual(len(stats["last_generation"]["itl_ms"]), 2)
        self.assertIsNotNone(stats["last_generation"]["mean_itl_ms"])
        self.assertGreaterEqual(stats["last_generation"]["ttft_ms"], 0.0)
        self.assertEqual(
            stats["last_generation"]["memory"]["peak_reserved_margin_bytes"],
            60,
        )

    def test_callback_synchronously_receives_every_sampled_token(self):
        eos = single_gpu.EOS_TOKEN_IDS[0]
        runner = _FakeRunner((31, 32, eos, 99))
        backend, _ = self._backend(runner, capacity=8)
        callbacks = []

        def token_callback(token):
            callbacks.append((token, backend.cache.consumed_len))

        try:
            generated = backend.generate_ids_stream(
                [1, 2],
                max_new_tokens=4,
                eos_token_ids=(eos,),
                token_callback=token_callback,
            )
        finally:
            backend.close()

        self.assertEqual(generated, [31, 32, eos])
        self.assertEqual(callbacks, [(31, 2), (32, 3), (eos, 4)])
        self.assertEqual(runner.reset_calls, 1)
        self.assertEqual(runner.cache.consumed_len, 0)
        self.assertEqual(backend.stats["last_generation"]["status"], "ok")
        self.assertEqual(backend.stats["last_generation"]["completion_tokens"], 3)

    def test_trace_uses_output_attribution_for_prefill_and_decode(self):
        runner = _FakeRunner((31, 32, 33))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            single_gpu, "ExpertTraceSession", _FakeTraceSession
        ):
            backend, _ = self._backend(runner, capacity=8, expert_trace_dir=directory)
            with mock.patch.object(
                backend, "_expert_trace_identity", return_value={"identity": "test"}
            ):
                try:
                    generated = backend.generate_ids(
                        [10, 11],
                        max_new_tokens=3,
                        eos_token_ids=(),
                        expert_trace=True,
                        request_id="cmpl-test",
                    )
                finally:
                    backend.close()

        self.assertEqual(generated, [31, 32, 33])
        self.assertEqual(len(_FakeTraceSession.instances), 1)
        session = _FakeTraceSession.instances[0]
        self.assertEqual(session.kwargs["request_id"], "cmpl-test")
        self.assertEqual(session.kwargs["max_rows"], 3)
        self.assertEqual(session.kwargs["prompt_tokens"], 2)
        self.assertEqual(
            [
                (step.phase, step.model_input_token_id, step.model_input_position)
                for step in session.steps
            ],
            [
                (single_gpu.PHASE_PREFILL_LAST, 11, 1),
                (single_gpu.PHASE_DECODE, 31, 2),
                (single_gpu.PHASE_DECODE, 32, 3),
            ],
        )
        self.assertEqual(
            [sampled_token for _, sampled_token in session.commits], [31, 32, 33]
        )
        self.assertEqual(
            [(phase, step) for phase, step in runner.trace_steps],
            [
                ("prefill", session.steps[0]),
                ("decode", session.steps[1]),
                ("decode", session.steps[2]),
            ],
        )
        self.assertEqual(
            session.finalize_calls,
            [{"status": "ok", "stopped_on_eos": False, "error": None}],
        )

    def test_trace_request_without_config_is_rejected_before_mutation(self):
        runner = _FakeRunner((31,))
        backend, _ = self._backend(runner, capacity=4)
        try:
            with self.assertRaisesRegex(ValueError, "no expert_trace_dir"):
                backend.generate_ids(
                    [1], max_new_tokens=1, expert_trace=True, request_id="test"
                )
        finally:
            backend.close()
        self.assertEqual(runner.embedded, [])
        self.assertEqual(backend.stats["request_count"], 0)

    def test_trace_finalize_failure_latches_backend(self):
        runner = _FakeRunner((31,))
        _FakeTraceSession.finalize_error = OSError("disk full")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            single_gpu, "ExpertTraceSession", _FakeTraceSession
        ):
            backend, _ = self._backend(runner, capacity=4, expert_trace_dir=directory)
            with mock.patch.object(backend, "_expert_trace_identity", return_value={}):
                try:
                    with self.assertRaisesRegex(OSError, "disk full"):
                        backend.generate_ids(
                            [1],
                            max_new_tokens=1,
                            expert_trace=True,
                            request_id="io-test",
                        )
                    self.assertTrue(backend.failed)
                    self.assertTrue(backend.cache.poisoned)
                    self.assertEqual(
                        backend.stats["last_generation"]["status"], "failed"
                    )
                    with self.assertRaisesRegex(RuntimeError, "restart"):
                        backend.generate_ids([2], max_new_tokens=1)
                finally:
                    backend.close()

    def test_callback_failure_resets_and_preserves_nonfatal_reuse(self):
        runner = _FakeRunner((31, 32))
        backend, _ = self._backend(runner, capacity=4)
        callbacks = []

        def token_callback(token):
            callbacks.append(token)
            raise RuntimeError("callback failed")

        try:
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                backend.generate_ids_stream(
                    [1], max_new_tokens=2, token_callback=token_callback
                )
            self.assertEqual(callbacks, [31])
            self.assertEqual(runner.reset_calls, 1)
            self.assertEqual(runner.cache.consumed_len, 0)
            self.assertFalse(runner.cache.poisoned)
            self.assertFalse(backend.failed)
            failed_stats = backend.stats["last_generation"]
            self.assertEqual(failed_stats["status"], "failed")
            self.assertEqual(failed_stats["completion_tokens"], 1)
            self.assertIn("callback failed", failed_stats["error"])

            self.assertEqual(backend.generate_ids([2], max_new_tokens=1), [32])
            self.assertEqual(runner.reset_calls, 2)
            self.assertEqual(backend.stats["request_count"], 2)
            self.assertEqual(backend.stats["last_generation"]["status"], "ok")
        finally:
            backend.close()

    def test_callback_is_validated_before_request_mutation(self):
        runner = _FakeRunner((31,))
        backend, _ = self._backend(runner, capacity=4)
        try:
            with self.assertRaisesRegex(TypeError, "token_callback must be callable"):
                backend.generate_ids_stream([1], max_new_tokens=1, token_callback=None)
        finally:
            backend.close()
        self.assertEqual(runner.embedded, [])
        self.assertEqual(runner.reset_calls, 0)
        self.assertEqual(backend.stats["request_count"], 0)

    def test_callback_and_reset_failure_poisons_backend(self):
        runner = _FakeRunner((31,))
        runner.reset_error = RuntimeError("cannot reset")
        backend, _ = self._backend(runner, capacity=4)

        def token_callback(token):
            raise RuntimeError("callback failed")

        try:
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                backend.generate_ids_stream(
                    [1], max_new_tokens=1, token_callback=token_callback
                )
            self.assertEqual(runner.reset_calls, 1)
            self.assertTrue(backend.failed)
            self.assertTrue(runner.cache.poisoned)
            self.assertEqual(backend.stats["last_generation"]["status"], "failed")
            with self.assertRaisesRegex(RuntimeError, "restart"):
                backend.generate_ids([2], max_new_tokens=1)
        finally:
            backend.close()

    def test_eos_stops_without_inserting_terminal_token_and_resets(self):
        eos = single_gpu.EOS_TOKEN_IDS[0]
        runner = _FakeRunner((eos, 99))
        backend, _ = self._backend(runner, capacity=8)
        try:
            self.assertEqual(
                backend.generate_ids([1, 2], max_new_tokens=4, eos_token_ids=(eos,)),
                [eos],
            )
        finally:
            backend.close()
        self.assertEqual(runner.decode_calls, [])
        self.assertEqual(runner.reset_calls, 1)

    def test_second_request_is_reported_as_warm(self):
        runner = _FakeRunner((41, 42))
        backend, _ = self._backend(runner, capacity=4)
        try:
            self.assertEqual(backend.generate_ids([1], max_new_tokens=1), [41])
            self.assertEqual(backend.stats["last_generation"]["profile"], "cold")
            self.assertEqual(backend.generate_ids([2], max_new_tokens=1), [42])
            self.assertEqual(backend.stats["last_generation"]["profile"], "warm")
            self.assertEqual(backend.stats["request_count"], 2)
        finally:
            backend.close()

    def test_validation_rejects_bad_ids_and_total_context_before_mutation(self):
        runner = _FakeRunner((1,))
        backend, _ = self._backend(runner, capacity=4)
        try:
            cases = (
                ([], 1, ValueError),
                ([True], 1, TypeError),
                ([single_gpu.TOKENIZER_VOCAB_SIZE], 1, ValueError),
                ([1, 2, 3, 4], 1, ValueError),
            )
            for prompt, count, error in cases:
                with self.subTest(prompt=prompt), self.assertRaises(error):
                    backend.generate_ids(prompt, max_new_tokens=count)
            self.assertEqual(backend.generate_ids([1, 2, 3, 4], max_new_tokens=0), [])
        finally:
            backend.close()
        self.assertEqual(runner.embedded, [])
        self.assertEqual(runner.reset_calls, 0)

    def test_regular_failure_resets_but_store_fatal_latches_backend(self):
        regular = _FakeRunner()
        regular.prefill_error = RuntimeError("ordinary kernel failure")
        backend, _ = self._backend(regular, capacity=4)
        with self.assertRaisesRegex(RuntimeError, "ordinary"):
            backend.generate_ids([1], max_new_tokens=1)
        self.assertEqual(regular.reset_calls, 1)
        self.assertFalse(backend.failed)
        backend.close()

        fatal = _FakeRunner()
        fatal.prefill_error = RuntimeError("checksum failed")
        fatal.fail_fatally = True
        failed_backend, _ = self._backend(fatal, capacity=4)
        try:
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                failed_backend.generate_ids([1], max_new_tokens=1)
            self.assertTrue(failed_backend.failed)
            self.assertTrue(failed_backend.cache.poisoned)
            self.assertEqual(fatal.reset_calls, 0)
            with self.assertRaisesRegex(RuntimeError, "restart"):
                failed_backend.generate_ids([2], max_new_tokens=1)
            self.assertEqual(fatal.embedded, [[1]])
        finally:
            failed_backend.close()

    def test_reset_failure_latches_backend_and_close_is_idempotent(self):
        runner = _FakeRunner((7,))
        runner.reset_error = RuntimeError("cannot reset")
        backend, _ = self._backend(runner, capacity=4)
        with self.assertRaisesRegex(RuntimeError, "cannot reset"):
            backend.generate_ids([1], max_new_tokens=1)
        self.assertTrue(backend.failed)
        self.assertTrue(backend.cache.poisoned)
        backend.close()
        backend.close()
        self.assertTrue(runner.closed)

    def test_device_contract_accepts_supported_sm70_and_sm89_cuda(self):
        for capability in ((7, 0), (8, 9)):
            with self.subTest(capability=capability), mock.patch.object(
                single_gpu.torch.cuda, "is_available", return_value=True
            ), mock.patch.object(
                single_gpu.torch.cuda, "device_count", return_value=1
            ), mock.patch.object(
                single_gpu.torch.cuda,
                "get_device_capability",
                return_value=capability,
            ):
                self.assertEqual(
                    single_gpu._require_single_supported_gpu(),
                    torch.device("cuda:0"),
                )

    def test_device_contract_rejects_absent_multiple_or_unsupported_cuda(self):
        cases = (
            ({"available": False, "count": 0, "capability": (7, 0)}, "CUDA"),
            ({"available": True, "count": 2, "capability": (7, 0)}, "exactly one"),
            ({"available": True, "count": 1, "capability": (8, 0)}, "SM70, SM89"),
        )
        for case, message in cases:
            with self.subTest(case=case), mock.patch.object(
                single_gpu.torch.cuda, "is_available", return_value=case["available"]
            ), mock.patch.object(
                single_gpu.torch.cuda, "device_count", return_value=case["count"]
            ), mock.patch.object(
                single_gpu.torch.cuda,
                "get_device_capability",
                return_value=case["capability"],
            ), self.assertRaisesRegex(
                RuntimeError, message
            ):
                single_gpu._require_single_supported_gpu()

    def test_device_uuid_falls_back_to_nvml_physical_index(self):
        properties = SimpleNamespace()
        with mock.patch.object(
            single_gpu.torch.cuda, "get_device_properties", return_value=properties
        ), mock.patch.object(
            single_gpu.torch.cuda, "_get_nvml_device_index", return_value=1
        ), mock.patch.object(
            single_gpu.torch.cuda,
            "_raw_device_uuid_nvml",
            return_value=["GPU-first", "GPU-second"],
        ):
            self.assertEqual(
                single_gpu._cuda_device_uuid(torch.device("cuda:0")), "GPU-second"
            )

    def test_cli_defaults_and_requested_offload_profile(self):
        defaults = single_gpu._parse_args([])
        self.assertEqual(defaults.expert_cache_mib, 7168)
        self.assertEqual(defaults.expert_stage_slots, 16)
        self.assertEqual(defaults.expert_io_workers, 2)
        self.assertEqual(defaults.capacity, 2048)

        configured = single_gpu._parse_args(
            [
                "--expert-pack-manifest",
                "/pack/manifest.json",
                "--expert-cache-mib",
                "6144",
                "--expert-stage-slots",
                "8",
                "--expert-io-workers",
                "1",
                "--capacity",
                "1024",
                "--raw-prompt",
                "--prompt",
                "Hello",
                "--max-new-tokens",
                "8",
                "--print-token-ids",
            ]
        )
        self.assertEqual(configured.expert_pack_manifest, "/pack/manifest.json")
        self.assertEqual(configured.expert_cache_mib, 6144)
        self.assertEqual(configured.capacity, 1024)
        self.assertTrue(configured.raw_prompt)
        self.assertTrue(configured.print_token_ids)

    def test_cli_raw_prompt_wires_backend_and_prints_tokens_and_stats(self):
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [10, 11]
        tokenizer.decode.return_value = "decoded"
        backend = mock.MagicMock()
        backend.__enter__.return_value = backend
        backend.__exit__.return_value = False
        backend.generate_ids.return_value = [31, 32]
        backend.stats = {"state": "READY"}
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            single_gpu, "load_tokenizer_compat", return_value=tokenizer
        ), mock.patch.object(
            single_gpu, "Qwen35SingleGPU", return_value=backend
        ) as backend_type, redirect_stdout(
            stdout
        ), redirect_stderr(
            stderr
        ):
            result = single_gpu.main(
                [
                    "--model-dir",
                    "/model",
                    "--expert-pack-manifest",
                    "/pack/manifest.json",
                    "--raw-prompt",
                    "--prompt",
                    "Hello",
                    "--max-new-tokens",
                    "2",
                    "--print-token-ids",
                ]
            )
        self.assertEqual(result, 0)
        tokenizer.encode.assert_called_once_with("Hello", add_special_tokens=False)
        backend.generate_ids.assert_called_once_with(
            [10, 11], max_new_tokens=2, eos_token_ids=single_gpu.EOS_TOKEN_IDS
        )
        _, backend_kwargs = backend_type.call_args
        self.assertEqual(backend_kwargs["expert_pack_manifest"], "/pack/manifest.json")
        self.assertEqual(stdout.getvalue(), "decoded\n")
        self.assertIn('QWEN35_SINGLE_GPU={"state": "READY"}', stderr.getvalue())
        self.assertIn("TOKEN_IDS=[31, 32]", stderr.getvalue())

    def test_cli_trace_output_sets_directory_and_request_basename(self):
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [10]
        tokenizer.decode.return_value = "decoded"
        backend = mock.MagicMock()
        backend.__enter__.return_value = backend
        backend.__exit__.return_value = False
        backend.generate_ids.return_value = [31]
        backend.stats = {"state": "READY"}
        with tempfile.TemporaryDirectory() as directory:
            output = f"{directory}/hello-trace"
            with mock.patch.object(
                single_gpu, "load_tokenizer_compat", return_value=tokenizer
            ), mock.patch.object(
                single_gpu, "Qwen35SingleGPU", return_value=backend
            ) as backend_type, redirect_stdout(
                io.StringIO()
            ), redirect_stderr(
                io.StringIO()
            ):
                result = single_gpu.main(
                    [
                        "--raw-prompt",
                        "--prompt",
                        "Hello",
                        "--max-new-tokens",
                        "1",
                        "--expert-trace-output",
                        output,
                    ]
                )
        self.assertEqual(result, 0)
        _, backend_kwargs = backend_type.call_args
        self.assertEqual(backend_kwargs["expert_trace_dir"], Path(directory))
        backend.generate_ids.assert_called_once_with(
            [10],
            max_new_tokens=1,
            eos_token_ids=single_gpu.EOS_TOKEN_IDS,
            expert_trace=True,
            request_id="hello-trace",
        )

    def test_cli_fatal_prints_stats_closes_and_propagates_for_nonzero_exit(self):
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [10]
        backend = mock.MagicMock()
        backend.__enter__.return_value = backend
        backend.__exit__.return_value = False
        backend.generate_ids.side_effect = RuntimeError("checksum mismatch")
        backend.stats = {"state": "FAILED"}
        stderr = io.StringIO()
        with mock.patch.object(
            single_gpu, "load_tokenizer_compat", return_value=tokenizer
        ), mock.patch.object(
            single_gpu, "Qwen35SingleGPU", return_value=backend
        ), redirect_stderr(
            stderr
        ), self.assertRaisesRegex(
            RuntimeError, "checksum mismatch"
        ):
            single_gpu.main(["--raw-prompt", "--prompt", "Hello"])
        self.assertIn('QWEN35_SINGLE_GPU={"state": "FAILED"}', stderr.getvalue())
        backend.__exit__.assert_called_once()

    def test_cli_stats_failure_does_not_replace_primary_generation_error(self):
        class BrokenStatsBackend:
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                self.closed = True

            @property
            def stats(self):
                raise RuntimeError("stats unavailable")

            def generate_ids(self, *args, **kwargs):
                raise RuntimeError("checksum mismatch")

        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [10]
        backend = BrokenStatsBackend()
        stderr = io.StringIO()
        with mock.patch.object(
            single_gpu, "load_tokenizer_compat", return_value=tokenizer
        ), mock.patch.object(
            single_gpu, "Qwen35SingleGPU", return_value=backend
        ), redirect_stderr(
            stderr
        ), self.assertRaisesRegex(
            RuntimeError, "checksum mismatch"
        ):
            single_gpu.main(["--raw-prompt", "--prompt", "Hello"])
        self.assertTrue(backend.closed)
        self.assertIn("stats unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
