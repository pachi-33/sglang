"""CPU-only HTTP contract tests for the Qwen3.5 dual-GPU pipeline API."""

import unittest

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


class _FakePipeline:
    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.calls = []
        self.closed = False
        self.error = None

    def generate_ids(self, prompt_ids, **kwargs):
        self.calls.append((list(prompt_ids), dict(kwargs)))
        if self.error is not None:
            raise self.error
        if not self.outputs:
            raise AssertionError("fake pipeline has no scripted output")
        return list(self.outputs.pop(0))

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

    def test_rejects_concurrency_non_greedy_streaming_and_wrong_model(self):
        fake, _, engine, app = self._fixture([71])
        with TestClient(app) as client:
            self.assertTrue(engine._request_lock.acquire(blocking=False))
            try:
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


if __name__ == "__main__":
    unittest.main()
