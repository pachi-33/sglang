"""CPU-only HTTP contract tests for the Qwen3.5 generation API."""

import json
import threading
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from sglang.srt.layers.qwen3_5 import pipeline_api
from sglang.srt.layers.qwen3_5.pipeline import EOS_TOKEN_IDS, PipelineWorkerError


class _FakeTokenizer:
    def __init__(self):
        self.raw_prompts = []
        self.chat_messages = []
        self.decoded = []

    def encode(self, prompt, *, add_special_tokens):
        self.raw_prompts.append((prompt, add_special_tokens))
        return [10, 11]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.chat_messages.append((messages, tokenize, add_generation_prompt))
        return [20, 21, 22]

    def decode(self, token_ids, *, skip_special_tokens):
        values = list(token_ids)
        self.decoded.append((values, skip_special_tokens))
        return "decoded:" + ",".join(str(token) for token in values)


class _BufferedByteTokenizer(_FakeTokenizer):
    def decode(self, token_ids, *, skip_special_tokens):
        values = list(token_ids)
        self.decoded.append((values, skip_special_tokens))
        if values == [101]:
            return "\ufffd"
        if values == [101, 102]:
            return "\u00e9"
        return super().decode(values, skip_special_tokens=skip_special_tokens)


class _FakePipeline:
    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.calls = []
        self.closed = False
        self.failed = False
        self.fail_fatally = False
        self.stats = {}
        self.error = None
        self.error_after_tokens = False
        self.stream_calls = []
        self.stream_started = threading.Event()
        self.stream_gate = None
        self.after_first_token_gate = None
        self.stream_finished = threading.Event()
        self.expert_trace_enabled = False

    def generate_ids(self, prompt_ids, **kwargs):
        self.calls.append((list(prompt_ids), dict(kwargs)))
        if self.error is not None:
            if self.fail_fatally:
                self.failed = True
            raise self.error
        if not self.outputs:
            raise AssertionError("fake pipeline has no scripted output")
        return list(self.outputs.pop(0))

    def generate_ids_stream(self, prompt_ids, *, token_callback, **kwargs):
        self.stream_calls.append((list(prompt_ids), dict(kwargs)))
        self.stream_started.set()
        if self.stream_gate is not None:
            self.stream_gate.wait(timeout=5)
        try:
            if self.error is not None and not self.error_after_tokens:
                if self.fail_fatally:
                    self.failed = True
                raise self.error
            if not self.outputs:
                raise AssertionError("fake pipeline has no scripted output")
            generated = list(self.outputs.pop(0))
            for index, token_id in enumerate(generated):
                token_callback(token_id)
                if index == 0 and self.after_first_token_gate is not None:
                    self.after_first_token_gate.wait(timeout=5)
            if self.error is not None:
                if self.fail_fatally:
                    self.failed = True
                raise self.error
            return generated
        finally:
            self.stream_finished.set()

    def close(self):
        self.closed = True


class TestPipelineAPI(unittest.TestCase):
    @staticmethod
    def _fixture(*outputs, api_key=None):
        fake_pipeline = _FakePipeline(outputs)
        tokenizer = _FakeTokenizer()
        engine = pipeline_api.Qwen35PipelineAPIEngine(
            fake_pipeline,
            tokenizer,
            model_id="agent-world",
        )
        app = pipeline_api.create_app(engine, api_key=api_key)
        return fake_pipeline, tokenizer, engine, app

    @staticmethod
    def _sse_payloads(response):
        values = []
        for line in response.text.splitlines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            values.append(data if data == "[DONE]" else json.loads(data))
        return values

    def test_native_generate_returns_text_tokens_usage_and_closes(self):
        fake, tokenizer, _, app = self._fixture([31, 32])
        with TestClient(app) as client:
            health = client.get("/health")
            response = client.post(
                "/generate",
                json={"input_ids": [1, 2], "max_new_tokens": 2},
            )
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["text"], "decoded:31,32")
        self.assertEqual(body["token_ids"], [31, 32])
        self.assertEqual(body["meta_info"]["prompt_tokens"], 2)
        self.assertEqual(body["meta_info"]["completion_tokens"], 2)
        self.assertEqual(body["meta_info"]["total_tokens"], 4)
        self.assertEqual(body["meta_info"]["finish_reason"], "length")
        self.assertEqual(
            fake.calls,
            [
                (
                    [1, 2],
                    {"max_new_tokens": 2, "eos_token_ids": EOS_TOKEN_IDS},
                )
            ],
        )
        self.assertEqual(tokenizer.decoded, [([31, 32], False)])
        self.assertTrue(fake.closed)

    def test_native_expert_trace_uses_response_id_as_local_basename(self):
        fake, _, _, app = self._fixture([31])
        fake.expert_trace_enabled = True
        with TestClient(app) as client:
            response = client.post(
                "/generate",
                json={
                    "input_ids": [1, 2],
                    "max_new_tokens": 1,
                    "expert_trace": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        request_id = response.json()["id"]
        self.assertTrue(request_id.startswith("gen-"))
        self.assertEqual(fake.calls[0][1]["expert_trace"], True)
        self.assertEqual(fake.calls[0][1]["request_id"], request_id)

    def test_expert_trace_requires_a_trace_capable_backend(self):
        fake, _, _, app = self._fixture([31])
        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "agent-world",
                    "prompt": "hello",
                    "max_tokens": 1,
                    "expert_trace": True,
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("no trace directory", response.json()["error"]["message"])
        self.assertEqual(fake.calls, [])

    def test_streaming_expert_trace_uses_sse_request_id(self):
        fake, _, _, app = self._fixture([41, 42])
        fake.expert_trace_enabled = True
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "agent-world",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 2,
                    "stream": True,
                    "expert_trace": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        payloads = self._sse_payloads(response)
        request_ids = {
            payload["id"]
            for payload in payloads
            if isinstance(payload, dict) and "id" in payload
        }
        self.assertEqual(len(request_ids), 1)
        request_id = request_ids.pop()
        self.assertTrue(request_id.startswith("chatcmpl-"))
        self.assertEqual(fake.stream_calls[0][1]["expert_trace"], True)
        self.assertEqual(fake.stream_calls[0][1]["request_id"], request_id)

    def test_completion_is_openai_shaped_and_omits_terminal_eos_from_text(self):
        eos = EOS_TOKEN_IDS[0]
        fake, tokenizer, _, app = self._fixture([41, eos])
        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "agent-world",
                    "prompt": "hello",
                    "max_tokens": 2,
                    "temperature": 0,
                },
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["model"], "agent-world")
        self.assertEqual(body["choices"][0]["text"], "decoded:41")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            body["usage"],
            {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        )
        self.assertEqual(body["sglang"]["completion_token_ids"], [41, eos])
        self.assertEqual(fake.calls[0][0], [10, 11])
        self.assertEqual(tokenizer.raw_prompts, [("hello", False)])

    def test_completion_ignore_eos_forces_fixed_length_benchmark_semantics(self):
        eos = EOS_TOKEN_IDS[0]
        fake, tokenizer, _, app = self._fixture([41, eos])
        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "agent-world",
                    "prompt": "hello",
                    "max_tokens": 2,
                    "temperature": 0,
                    "best_of": 1,
                    "stream": False,
                    "ignore_eos": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["choices"][0]["text"], f"decoded:41,{eos}")
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(
            fake.calls,
            [
                (
                    [10, 11],
                    {"max_new_tokens": 2, "eos_token_ids": ()},
                )
            ],
        )
        self.assertEqual(tokenizer.decoded, [([41, eos], False)])

    def test_completion_stream_emits_tokens_finish_usage_and_done(self):
        eos = EOS_TOKEN_IDS[0]
        fake, _, _, app = self._fixture([41, 42, eos])
        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "agent-world",
                    "prompt": "hello",
                    "max_tokens": 3,
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream")
        )
        chunks = self._sse_payloads(response)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["object"], "text_completion")
        self.assertEqual(chunks[0]["choices"][0]["text"], "decoded:41")
        self.assertIsNone(chunks[0]["choices"][0]["finish_reason"])
        self.assertEqual(chunks[1]["choices"][0]["text"], ",42")
        self.assertEqual(chunks[2]["choices"][0]["text"], "")
        self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            chunks[2]["usage"],
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )
        self.assertEqual(chunks[2]["sglang"]["completion_token_ids"], [41, 42, eos])
        self.assertEqual(chunks[3], "[DONE]")
        self.assertEqual(
            fake.stream_calls,
            [([10, 11], {"max_new_tokens": 3, "eos_token_ids": EOS_TOKEN_IDS})],
        )

    def test_completion_stream_buffers_incomplete_utf8_without_losing_token_event(self):
        fake = _FakePipeline(([101, 102],))
        tokenizer = _BufferedByteTokenizer()
        engine = pipeline_api.Qwen35PipelineAPIEngine(
            fake,
            tokenizer,
            model_id="agent-world",
        )
        app = pipeline_api.create_app(engine)
        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "agent-world",
                    "prompt": [1],
                    "max_tokens": 2,
                    "stream": True,
                },
            )

        chunks = self._sse_payloads(response)
        token_chunks = chunks[:2]
        self.assertEqual(
            [chunk["sglang"]["completion_token_ids"] for chunk in token_chunks],
            [[101], [102]],
        )
        self.assertEqual(
            "".join(chunk["choices"][0]["text"] for chunk in chunks[:-1]),
            "\u00e9",
        )
        self.assertEqual(chunks[0]["choices"][0]["text"], "")
        self.assertEqual(chunks[1]["choices"][0]["text"], "\u00e9")
        self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(chunks[3], "[DONE]")

    def test_chat_applies_checkpoint_template_and_supports_current_token_field(self):
        fake, tokenizer, _, app = self._fixture([51, 52])
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "agent-world",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 2,
                    "temperature": 0,
                },
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(
            body["choices"][0]["message"],
            {"role": "assistant", "content": "decoded:51,52"},
        )
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["prompt_tokens"], 3)
        self.assertEqual(
            tokenizer.chat_messages,
            [([{"role": "user", "content": "hello"}], True, True)],
        )
        self.assertEqual(fake.calls[0][0], [20, 21, 22])

    def test_chat_stream_starts_with_role_and_ends_with_finish_and_done(self):
        fake, _, _, app = self._fixture([51, 52])
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "agent-world",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 2,
                    "stream": True,
                },
            )

        chunks = self._sse_payloads(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(
            chunks[0]["choices"][0]["delta"],
            {"role": "assistant", "content": ""},
        )
        self.assertEqual(chunks[1]["choices"][0]["delta"]["content"], "decoded:51")
        self.assertEqual(chunks[2]["choices"][0]["delta"]["content"], ",52")
        self.assertEqual(chunks[3]["choices"][0]["delta"], {})
        self.assertEqual(chunks[3]["choices"][0]["finish_reason"], "length")
        self.assertEqual(chunks[3]["usage"]["completion_tokens"], 2)
        self.assertEqual(chunks[4], "[DONE]")
        self.assertEqual(len({chunk["id"] for chunk in chunks[:-1]}), 1)

    def test_api_key_guards_model_and_generation_routes(self):
        _, _, _, app = self._fixture([61], api_key="secret")
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            unauthorized = client.get("/v1/models")
            models = client.get(
                "/v1/models", headers={"Authorization": "Bearer secret"}
            )
            completion = client.post(
                "/v1/completions",
                headers={"Authorization": "Bearer secret"},
                json={"model": "agent-world", "prompt": [1], "max_tokens": 1},
            )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized.headers["www-authenticate"], "Bearer")
        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json()["data"][0]["id"], "agent-world")
        self.assertEqual(completion.status_code, 200)

    def test_rejects_concurrency_non_greedy_legacy_streaming_and_wrong_model(self):
        fake, _, engine, app = self._fixture([71])
        fake.generate_ids_stream = None
        with TestClient(app) as client:
            self.assertTrue(engine._request_lock.acquire(blocking=False))
            try:
                busy_health = client.get("/health")
                busy = client.post(
                    "/v1/completions",
                    json={"model": "agent-world", "prompt": "x"},
                )
            finally:
                engine._request_lock.release()
            non_greedy = client.post(
                "/v1/completions",
                json={
                    "model": "agent-world",
                    "prompt": "x",
                    "temperature": 0.5,
                },
            )
            streaming = client.post(
                "/v1/chat/completions",
                json={
                    "model": "agent-world",
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
            boolean_temperature = client.post(
                "/v1/completions",
                json={
                    "model": "agent-world",
                    "prompt": "x",
                    "temperature": True,
                },
            )
            wrong_model = client.post(
                "/v1/completions",
                json={"model": "other", "prompt": "x"},
            )
        self.assertEqual(busy.status_code, 429)
        self.assertEqual(busy.json()["error"]["type"], "server_busy")
        self.assertTrue(busy_health.json()["busy"])
        self.assertEqual(non_greedy.status_code, 400)
        self.assertIn("greedy", non_greedy.json()["error"]["message"])
        self.assertEqual(streaming.status_code, 400)
        self.assertIn("streaming", streaming.json()["error"]["message"])
        self.assertEqual(boolean_temperature.status_code, 400)
        self.assertEqual(wrong_model.status_code, 404)
        self.assertEqual(fake.calls, [])

    def test_native_requires_exactly_one_input_and_strict_integer_ids(self):
        _, _, _, app = self._fixture()
        with TestClient(app) as client:
            missing = client.post("/generate", json={"max_new_tokens": 1})
            multiple = client.post(
                "/generate",
                json={"text": "x", "input_ids": [1], "max_new_tokens": 1},
            )
            coerced = client.post(
                "/generate",
                json={"input_ids": [True], "max_new_tokens": 1},
            )
            empty_messages = client.post(
                "/generate",
                json={"messages": [], "max_new_tokens": 1},
            )
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(multiple.status_code, 400)
        self.assertEqual(coerced.status_code, 400)
        self.assertEqual(empty_messages.status_code, 400)

    def test_worker_failures_are_reported_without_leaking_success(self):
        fake, _, _, app = self._fixture()
        fake.error = PipelineWorkerError("front worker failed")
        with self.assertLogs(pipeline_api.logger.name, level="ERROR"):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/completions",
                    json={
                        "model": "agent-world",
                        "prompt": [1],
                        "max_tokens": 1,
                    },
                )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["type"], "worker_error")
        self.assertEqual(len(fake.calls), 1)

    def test_stream_holds_slot_until_backend_cleanup_and_rejects_concurrency(self):
        fake, _, engine, app = self._fixture([81])
        fake.stream_gate = threading.Event()
        stream = engine.stream_raw([1], 1)
        self.assertTrue(fake.stream_started.wait(timeout=1))
        self.assertTrue(engine.busy)

        with TestClient(app) as client:
            busy = client.post(
                "/v1/completions",
                json={"model": "agent-world", "prompt": [2], "max_tokens": 1},
            )
            self.assertEqual(busy.status_code, 429)
            self.assertEqual(busy.json()["error"]["type"], "server_busy")
            fake.stream_gate.set()
            terminal = list(stream)[-1]

        self.assertIsInstance(terminal, pipeline_api._StreamTerminal)
        self.assertIsNone(terminal.error)
        self.assertTrue(fake.stream_finished.is_set())
        self.assertFalse(engine.busy)

    def test_stream_error_emits_error_then_done_and_close_waits_for_worker(self):
        fake, _, engine, app = self._fixture([101])
        fake.error = RuntimeError("generation exploded")
        fake.error_after_tokens = True
        with self.assertLogs(pipeline_api.logger.name, level="ERROR"):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/completions",
                    json={
                        "model": "agent-world",
                        "prompt": [1],
                        "max_tokens": 1,
                        "stream": True,
                    },
                )
        chunks = self._sse_payloads(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(chunks[0]["choices"][0]["text"], "decoded:101")
        self.assertEqual(chunks[1]["error"]["type"], "server_error")
        self.assertIn("generation exploded", chunks[1]["error"]["message"])
        self.assertEqual(chunks[2], "[DONE]")
        self.assertTrue(fake.stream_finished.is_set())
        self.assertTrue(fake.closed)
        self.assertFalse(engine.busy)

        immediate, _, _, immediate_app = self._fixture()
        immediate.error = PipelineWorkerError("prefill failed")
        with self.assertLogs(pipeline_api.logger.name, level="ERROR"):
            with TestClient(immediate_app) as client:
                immediate_response = client.post(
                    "/v1/completions",
                    json={
                        "model": "agent-world",
                        "prompt": [1],
                        "max_tokens": 1,
                        "stream": True,
                    },
                )
        self.assertEqual(immediate_response.status_code, 503)
        self.assertEqual(immediate_response.json()["error"]["type"], "worker_error")

        disconnected, _, disconnected_engine, _ = self._fixture([111, 112])
        disconnected.after_first_token_gate = threading.Event()
        disconnected_stream = disconnected_engine.stream_raw([1], 2)
        first_event = disconnected_stream.prefetch()
        self.assertIsInstance(first_event, pipeline_api._StreamToken)
        disconnected_stream.cancel()
        disconnected.after_first_token_gate.set()
        disconnect_terminal = list(disconnected_stream)[-1]
        self.assertIn("disconnected", str(disconnect_terminal.error))
        self.assertTrue(disconnected.stream_finished.is_set())
        self.assertFalse(disconnected_engine.busy)
        disconnected_engine.close()

        blocking, _, blocking_engine, _ = self._fixture([91])
        blocking.stream_gate = threading.Event()
        blocking_engine.stream_raw([1], 1)
        self.assertTrue(blocking.stream_started.wait(timeout=1))
        close_started = threading.Event()

        def close_engine():
            close_started.set()
            blocking_engine.close()

        close_thread = threading.Thread(target=close_engine)
        close_thread.start()
        self.assertTrue(close_started.wait(timeout=1))
        close_thread.join(timeout=0.05)
        self.assertTrue(close_thread.is_alive())
        self.assertFalse(blocking.closed)
        blocking.stream_gate.set()
        close_thread.join(timeout=1)
        self.assertFalse(close_thread.is_alive())
        self.assertTrue(blocking.stream_finished.is_set())
        self.assertTrue(blocking.closed)

    def test_backend_fatal_returns_503_and_health_remains_failed(self):
        fake, _, _, app = self._fixture()
        fake.error = RuntimeError("ExpertPack checksum mismatch")
        fake.fail_fatally = True
        fake.stats = {"state": "FAILED", "checksum_failures": 1}
        with self.assertLogs(pipeline_api.logger.name, level="ERROR"):
            with TestClient(app) as client:
                first = client.post(
                    "/generate", json={"input_ids": [1], "max_new_tokens": 1}
                )
                health = client.get("/health")
                second = client.post(
                    "/v1/completions",
                    json={
                        "model": "not-served",
                        "prompt": [2],
                        "max_tokens": 1,
                        "temperature": 0.5,
                    },
                )
        self.assertEqual(first.status_code, 503)
        self.assertEqual(first.json()["error"]["type"], "backend_failed")
        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["status"], "failed")
        self.assertEqual(health.json()["stats"]["checksum_failures"], 1)
        self.assertEqual(second.status_code, 503)
        self.assertEqual(second.json()["error"]["type"], "backend_failed")
        self.assertEqual(len(fake.calls), 1)

    def test_failed_health_survives_broken_stats_snapshot(self):
        fake, _, _, app = self._fixture()
        fake.failed = True
        with mock.patch.object(
            type(fake),
            "stats",
            new_callable=mock.PropertyMock,
            side_effect=RuntimeError("snapshot unavailable"),
            create=True,
        ):
            with TestClient(app) as client:
                health = client.get("/health")
        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["status"], "failed")
        self.assertIn("snapshot unavailable", health.json()["stats_error"])

    def test_server_cli_exposes_front_back_and_split_layout(self):
        defaults = pipeline_api._parse_args([])
        self.assertIsNone(defaults.front_uuid)
        self.assertIsNone(defaults.back_uuid)
        self.assertEqual(defaults.split_layer, 17)

        configured = pipeline_api._parse_args(
            [
                "--front-uuid",
                "GPU-front",
                "--back-uuid",
                "GPU-back",
                "--split-layer",
                "13",
            ]
        )
        self.assertEqual(configured.front_uuid, "GPU-front")
        self.assertEqual(configured.back_uuid, "GPU-back")
        self.assertEqual(configured.split_layer, 13)


if __name__ == "__main__":
    unittest.main()
