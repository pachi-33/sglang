import argparse
import asyncio
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang import bench_serving
from sglang.bench_serving import (
    RequestFuncInput,
    RequestFuncOutput,
    _iter_sse_data,
    async_request_openai_completions,
    benchmark,
    get_tokenizer,
    positive_int,
    request_with_concurrency_limit,
    sample_random_requests,
)


class _AsyncByteChunks:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks, *, status=200, reason="OK"):
        self.content = _AsyncByteChunks(chunks)
        self.status = status
        self.reason = reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, *, url, json, headers):
        self.requests.append((url, json, headers))
        return self.response


class TestSSEParsing(unittest.IsolatedAsyncioTestCase):
    async def test_frames_survive_split_coalesced_and_multiline_data(self):
        wire = (
            b": keepalive\r\n"
            b"event: message\r\n"
            b'data: {"choices":\r\n'
            b'data: [{"text":"first"}]}\r\n\r\n'
            b'data: {"choices":[{"text":"second"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        cuts = (1, 8, 19, 37, 64, len(wire) - 5)
        chunks = []
        start = 0
        for end in cuts:
            chunks.append(wire[start:end])
            start = end
        chunks.append(wire[start:])

        payloads = [
            payload async for payload in _iter_sse_data(_AsyncByteChunks(chunks))
        ]

        self.assertEqual(len(payloads), 3)
        self.assertEqual(json.loads(payloads[0])["choices"][0]["text"], "first")
        self.assertEqual(json.loads(payloads[1])["choices"][0]["text"], "second")
        self.assertEqual(payloads[2], "[DONE]")

    async def test_completion_timing_excludes_ttft_from_itl(self):
        events = (
            b'data: {"choices":[{"text":"a","finish_reason":null}],'
            b'"sglang":{"completion_token_ids":[1]}}\n\n'
            b'data: {"choices":[{"text":"","finish_reason":null}],'
            b'"sglang":{"completion_token_ids":[2]}}\n\n'
            b'data: {"choices":[{"text":"c","finish_reason":null}],'
            b'"sglang":{"completion_token_ids":[3]}}\n\n'
            b'data: {"choices":[{"text":"","finish_reason":"length"}],'
            b'"usage":{"completion_tokens":3},'
            b'"sglang":{"completion_token_ids":[1,2,3]}}\n\n'
            b"data: [DONE]\n\n"
        )
        # Split inside the first record and coalesce all later records.
        response = _FakeResponse((events[:11], events[11:]))
        session = _FakeSession(response)
        request = RequestFuncInput(
            prompt="prompt",
            api_url="http://127.0.0.1:30000/v1/completions",
            prompt_len=4,
            output_len=3,
            model="model",
        )
        # start, three text events, usage event, DONE
        timestamps = (100.0, 102.0, 105.0, 109.0, 109.5, 110.0)
        with mock.patch.object(
            bench_serving.aiohttp, "ClientSession", return_value=session
        ), mock.patch.object(
            bench_serving, "args", SimpleNamespace(disable_stream=False), create=True
        ), mock.patch.object(
            bench_serving.time, "perf_counter", side_effect=timestamps
        ):
            output = await async_request_openai_completions(request)

        self.assertTrue(output.success, output.error)
        self.assertEqual(output.generated_text, "ac")
        self.assertEqual(output.output_len, 3)
        self.assertEqual(output.ttft, 2.0)
        self.assertEqual(output.itl, [3.0, 4.0])
        self.assertEqual(len(output.itl), output.output_len - 1)
        self.assertEqual(output.latency, 10.0)
        self.assertTrue(session.requests[0][1]["stream"])
        self.assertTrue(session.requests[0][1]["ignore_eos"])
        self.assertNotIn("expert_trace", session.requests[0][1])

    async def test_traced_token_id_request_captures_identity_and_prompt_usage(self):
        events = (
            b'data: {"id":"cmpl-trace","choices":[{"text":"a",'
            b'"finish_reason":null}],"sglang":{"completion_token_ids":[7]}}\n\n'
            b'data: {"id":"cmpl-trace","choices":[{"text":"",'
            b'"finish_reason":"length"}],"usage":{"prompt_tokens":4,'
            b'"completion_tokens":1}}\n\n'
            b"data: [DONE]\n\n"
        )
        response = _FakeResponse((events,))
        session = _FakeSession(response)
        request = RequestFuncInput(
            prompt=[1, 2, 3, 4],
            api_url="http://127.0.0.1:30000/v1/completions",
            prompt_len=4,
            output_len=1,
            model="model",
            expert_trace=True,
        )
        with mock.patch.object(
            bench_serving.aiohttp, "ClientSession", return_value=session
        ), mock.patch.object(
            bench_serving, "args", SimpleNamespace(disable_stream=False), create=True
        ), mock.patch.object(
            bench_serving.time,
            "perf_counter",
            side_effect=(100.0, 101.0, 102.0, 103.0),
        ):
            output = await async_request_openai_completions(request)

        self.assertTrue(output.success, output.error)
        self.assertEqual(output.request_id, "cmpl-trace")
        self.assertEqual(output.reported_prompt_len, 4)
        self.assertEqual(session.requests[0][1]["prompt"], [1, 2, 3, 4])
        self.assertIs(session.requests[0][1]["expert_trace"], True)

    async def test_traced_request_rejects_prompt_length_mismatch(self):
        events = (
            b'data: {"id":"cmpl-trace","choices":[{"text":"a",'
            b'"finish_reason":null}],"sglang":{"completion_token_ids":[7]}}\n\n'
            b'data: {"id":"cmpl-trace","choices":[{"text":"",'
            b'"finish_reason":"length"}],"usage":{"prompt_tokens":3,'
            b'"completion_tokens":1}}\n\n'
            b"data: [DONE]\n\n"
        )
        response = _FakeResponse((events,))
        request = RequestFuncInput(
            prompt=[1, 2, 3, 4],
            api_url="http://127.0.0.1:30000/v1/completions",
            prompt_len=4,
            output_len=1,
            model="model",
            expert_trace=True,
        )
        with mock.patch.object(
            bench_serving.aiohttp,
            "ClientSession",
            return_value=_FakeSession(response),
        ), mock.patch.object(
            bench_serving, "args", SimpleNamespace(disable_stream=False), create=True
        ), mock.patch.object(
            bench_serving.time,
            "perf_counter",
            side_effect=(100.0, 101.0, 102.0, 103.0),
        ):
            output = await async_request_openai_completions(request)

        self.assertFalse(output.success)
        self.assertIn("3 != 4", output.error)

    async def test_non_streaming_raw_json_remains_supported(self):
        body = (
            b'{"choices":[{"text":"whole response","finish_reason":"length"}],'
            b'"usage":{"completion_tokens":2}}'
        )
        response = _FakeResponse((body[:9], body[9:]))
        session = _FakeSession(response)
        request = RequestFuncInput(
            prompt="prompt",
            api_url="http://127.0.0.1:30000/v1/completions",
            prompt_len=4,
            output_len=2,
            model="model",
        )
        with mock.patch.object(
            bench_serving.aiohttp, "ClientSession", return_value=session
        ), mock.patch.object(
            bench_serving, "args", SimpleNamespace(disable_stream=True), create=True
        ), mock.patch.object(
            bench_serving.time, "perf_counter", side_effect=(100.0, 105.0)
        ):
            output = await async_request_openai_completions(request)

        self.assertTrue(output.success, output.error)
        self.assertEqual(output.generated_text, "whole response")
        self.assertEqual(output.ttft, 5.0)
        self.assertEqual(output.itl, [])
        self.assertEqual(output.latency, 5.0)
        self.assertFalse(session.requests[0][1]["stream"])

    async def test_streaming_early_eof_is_not_reported_as_success(self):
        response = _FakeResponse((b'data: {"choices":[{"text":"a"}]}\n\n',))
        session = _FakeSession(response)
        request = RequestFuncInput(
            prompt="prompt",
            api_url="http://127.0.0.1:30000/v1/completions",
            prompt_len=4,
            output_len=1,
            model="model",
        )
        with mock.patch.object(
            bench_serving.aiohttp, "ClientSession", return_value=session
        ), mock.patch.object(
            bench_serving, "args", SimpleNamespace(disable_stream=False), create=True
        ), mock.patch.object(
            bench_serving.time, "perf_counter", side_effect=(100.0, 102.0)
        ):
            output = await async_request_openai_completions(request)

        self.assertFalse(output.success)
        self.assertIn("[DONE]", output.error)

    async def test_streaming_done_without_token_is_not_success(self):
        response = _FakeResponse((b"data: [DONE]\n\n",))
        session = _FakeSession(response)
        request = RequestFuncInput(
            prompt="prompt",
            api_url="http://127.0.0.1:30000/v1/completions",
            prompt_len=4,
            output_len=1,
            model="model",
        )
        with mock.patch.object(
            bench_serving.aiohttp, "ClientSession", return_value=session
        ), mock.patch.object(
            bench_serving, "args", SimpleNamespace(disable_stream=False), create=True
        ), mock.patch.object(
            bench_serving.time, "perf_counter", side_effect=(100.0, 101.0)
        ):
            output = await async_request_openai_completions(request)

        self.assertFalse(output.success)
        self.assertIn("no output token", output.error)

    async def test_terminal_usage_must_match_fixed_output_length(self):
        events = (
            b'data: {"choices":[{"text":"a","finish_reason":null}]}\n\n'
            b'data: {"choices":[{"text":"","finish_reason":"length"}],'
            b'"usage":{"completion_tokens":2}}\n\n'
            b"data: [DONE]\n\n"
        )
        response = _FakeResponse((events,))
        session = _FakeSession(response)
        request = RequestFuncInput(
            prompt="prompt",
            api_url="http://127.0.0.1:30000/v1/completions",
            prompt_len=4,
            output_len=1,
            model="model",
        )
        with mock.patch.object(
            bench_serving.aiohttp, "ClientSession", return_value=session
        ), mock.patch.object(
            bench_serving, "args", SimpleNamespace(disable_stream=False), create=True
        ), mock.patch.object(
            bench_serving.time,
            "perf_counter",
            side_effect=(100.0, 101.0, 102.0, 103.0),
        ):
            output = await async_request_openai_completions(request)

        self.assertFalse(output.success)
        self.assertIn("2 != 1", output.error)


class TestBenchServingConcurrency(unittest.IsolatedAsyncioTestCase):
    async def test_semaphore_limits_in_flight_requests(self):
        active = 0
        peak = 0

        async def request_func(request_func_input, pbar=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return RequestFuncOutput(success=True)

        request = RequestFuncInput("prompt", "url", 1, 1, "model")
        semaphore = asyncio.Semaphore(2)
        outputs = await asyncio.gather(
            *(
                request_with_concurrency_limit(
                    request_func,
                    request,
                    None,
                    semaphore,
                )
                for _ in range(6)
            )
        )

        self.assertEqual(peak, 2)
        self.assertTrue(all(output.success for output in outputs))

    async def test_none_keeps_requests_unlimited(self):
        started = 0
        all_started = asyncio.Event()

        async def request_func(request_func_input, pbar=None):
            nonlocal started
            started += 1
            if started == 4:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            return RequestFuncOutput(success=True)

        request = RequestFuncInput("prompt", "url", 1, 1, "model")
        await asyncio.gather(
            *(
                request_with_concurrency_limit(request_func, request, None, None)
                for _ in range(4)
            )
        )

        self.assertEqual(started, 4)

    async def test_benchmark_traces_main_requests_but_not_initial_probe(self):
        calls = []

        async def request_func(request_func_input, pbar=None):
            calls.append(request_func_input)
            if pbar is not None:
                pbar.update(1)
            return RequestFuncOutput(
                generated_text="ab",
                success=True,
                latency=1.0,
                ttft=0.25,
                itl=[0.75],
                prompt_len=request_func_input.prompt_len,
                output_len=request_func_input.output_len,
                request_id=(
                    f"cmpl-{request_func_input.request_index}"
                    if request_func_input.expert_trace
                    else ""
                ),
                reported_prompt_len=(
                    request_func_input.prompt_len
                    if request_func_input.expert_trace
                    else None
                ),
            )

        class Tokenizer:
            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[1, 2])

        requests = [([1, 2], 2, 2), ([3, 4], 2, 2)]
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            bench_serving.ASYNC_REQUEST_FUNCS, {"sglang": request_func}, clear=True
        ), mock.patch.object(
            bench_serving,
            "args",
            SimpleNamespace(
                backend="sglang",
                dataset_name="random",
                request_rate=float("inf"),
                sharegpt_output_len=None,
                random_input_len=2,
                random_output_len=2,
                random_range_ratio=1.0,
                num_prompts=2,
                output_file=str(Path(directory) / "bench.jsonl"),
            ),
            create=True,
        ):
            await benchmark(
                backend="sglang",
                api_url="http://127.0.0.1:30000/v1/completions",
                model_id="model",
                tokenizer=Tokenizer(),
                input_requests=requests,
                request_rate=float("inf"),
                disable_tqdm=True,
                enable_multi=False,
                max_concurrency=1,
                expert_trace=True,
            )
            record = json.loads(
                (Path(directory) / "bench.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual([call.expert_trace for call in calls], [False, True, True])
        self.assertEqual(
            [item["trace_id"] for item in record["trace_requests"]],
            ["cmpl-0", "cmpl-1"],
        )
        self.assertTrue(
            all(
                item["prompt_kind"] == "token_ids_int32_le"
                for item in record["trace_requests"]
            )
        )


class TestPositiveInt(unittest.TestCase):
    def test_accepts_positive_integer(self):
        self.assertEqual(positive_int("3"), 3)

    def test_rejects_non_positive_or_non_integer(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_int(value)


class TestTokenizerCompatibility(unittest.TestCase):
    def test_qwen35_local_checkpoint_uses_compat_loader(self):
        tokenizer = object()
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5_moe"}), encoding="utf-8"
            )
            with mock.patch(
                "sglang.srt.layers.qwen3_5.pipeline.load_tokenizer_compat",
                return_value=tokenizer,
            ) as loader:
                self.assertIs(get_tokenizer(str(model_dir)), tokenizer)
        loader.assert_called_once_with(str(model_dir))


class TestRandomTokenIdRequests(unittest.TestCase):
    def test_return_token_ids_preserves_exact_requested_lengths(self):
        class Tokenizer:
            def __call__(self, text):
                return SimpleNamespace(input_ids=[len(text), 2, 3])

            def decode(self, token_ids):
                return "decoded"

        dataset = [
            {
                "conversations": [
                    {"value": "first"},
                    {"value": "answer"},
                ]
            },
            {
                "conversations": [
                    {"value": "second"},
                    {"value": "answer"},
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            random.seed(0)
            bench_serving.np.random.seed(0)
            requests = sample_random_requests(
                input_len=5,
                output_len=4,
                num_prompts=2,
                range_ratio=1.0,
                tokenizer=Tokenizer(),
                dataset_path=str(path),
                return_token_ids=True,
            )

        self.assertEqual([len(prompt) for prompt, _, _ in requests], [5, 5])
        self.assertEqual([prompt_len for _, prompt_len, _ in requests], [5, 5])
        self.assertTrue(all(isinstance(prompt, list) for prompt, _, _ in requests))


if __name__ == "__main__":
    unittest.main()
