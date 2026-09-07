"""HTTP API for the fixed two-GPU Qwen3.5 single-request pipeline.

This adapter deliberately preserves the pipeline's narrow execution contract:
one active request, batch one, greedy decoding, and non-streaming responses.
It exposes a small native ``/generate`` endpoint plus non-streaming OpenAI-style
completion and chat-completion endpoints.  It does not route through SGLang's
normal scheduler, radix cache, or general server state.
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from starlette.concurrency import run_in_threadpool

from .pipeline import (
    EOS_TOKEN_IDS,
    MODEL_DIR_DEFAULT,
    SM89_UUID,
    V100_UUID,
    PipelineProtocolError,
    PipelineWorkerError,
    Qwen35Pipeline,
    load_tokenizer_compat,
)

logger = logging.getLogger(__name__)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
StrictNumber = StrictInt | StrictFloat


class PipelineBusyError(RuntimeError):
    """Raised when a second request arrives while the sole slot is active."""


class APIRequestError(ValueError):
    """A client error with an explicit HTTP status."""

    def __init__(self, message: str, status_code: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status_code = int(status_code)


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessageRequest(_RequestModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None = None


class TextResponseFormat(_RequestModel):
    type: Literal["text"] = "text"


class CompletionRequest(_RequestModel):
    model: str
    prompt: str | list[StrictInt]
    max_tokens: StrictInt = Field(default=16, ge=1, le=2048)
    n: StrictInt = Field(default=1, ge=1)
    stream: StrictBool = False
    temperature: StrictNumber | None = 0.0
    top_p: StrictNumber | None = 1.0
    stop: str | list[str] | None = None
    frequency_penalty: StrictNumber | None = 0.0
    presence_penalty: StrictNumber | None = 0.0
    echo: StrictBool = False
    best_of: StrictInt | None = None
    logprobs: StrictInt | None = None
    logit_bias: dict[str, float] | None = None
    seed: StrictInt | None = None
    suffix: str | None = None
    user: str | None = None


class ChatCompletionRequest(_RequestModel):
    model: str
    messages: list[ChatMessageRequest] = Field(min_length=1)
    max_tokens: StrictInt | None = Field(default=None, ge=1, le=2048)
    max_completion_tokens: StrictInt | None = Field(default=None, ge=1, le=2048)
    n: StrictInt = Field(default=1, ge=1)
    stream: StrictBool = False
    temperature: StrictNumber | None = 0.0
    top_p: StrictNumber | None = 1.0
    stop: str | list[str] | None = None
    frequency_penalty: StrictNumber | None = 0.0
    presence_penalty: StrictNumber | None = 0.0
    logprobs: StrictBool | None = False
    top_logprobs: StrictInt | None = None
    logit_bias: dict[str, float] | None = None
    response_format: TextResponseFormat | None = None
    seed: StrictInt | None = None
    user: str | None = None


class NativeGenerateRequest(_RequestModel):
    text: str | None = None
    input_ids: list[StrictInt] | None = None
    messages: list[ChatMessageRequest] | None = None
    max_new_tokens: StrictInt = Field(default=16, ge=1, le=2048)


@dataclass(frozen=True)
class PipelineGeneration:
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    text: str
    finish_reason: Literal["stop", "length"]


class Qwen35PipelineAPIEngine:
    """Serialize HTTP calls onto the pipeline's one reusable request slot."""

    def __init__(
        self,
        pipeline: Qwen35Pipeline,
        tokenizer: Any,
        *,
        model_id: str,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._request_lock = threading.Lock()
        self._closed = False

    @property
    def busy(self) -> bool:
        return self._request_lock.locked()

    @property
    def closed(self) -> bool:
        return self._closed

    def _generate_locked(
        self,
        prompt_builder: Callable[[], Sequence[int]],
        max_new_tokens: int,
    ) -> PipelineGeneration:
        if not self._request_lock.acquire(blocking=False):
            raise PipelineBusyError("the single pipeline request slot is busy")
        try:
            if self._closed:
                raise RuntimeError("the pipeline API engine is closed")
            prompt_ids = list(prompt_builder())
            if not prompt_ids:
                raise APIRequestError("prompt must produce at least one token")
            generated = self.pipeline.generate_ids(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=EOS_TOKEN_IDS,
            )
            stopped = bool(generated and generated[-1] in EOS_TOKEN_IDS)
            visible_ids = [token for token in generated if token not in EOS_TOKEN_IDS]
            return PipelineGeneration(
                prompt_token_ids=tuple(prompt_ids),
                completion_token_ids=tuple(generated),
                text=self.tokenizer.decode(visible_ids, skip_special_tokens=False),
                finish_reason="stop" if stopped else "length",
            )
        finally:
            self._request_lock.release()

    def complete_raw(
        self, prompt: str | Sequence[int], max_new_tokens: int
    ) -> PipelineGeneration:
        if isinstance(prompt, str):
            if not prompt:
                raise APIRequestError("prompt must not be empty")
            builder = lambda: self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_ids = list(prompt)
            builder = lambda: prompt_ids
        return self._generate_locked(builder, max_new_tokens)

    def complete_chat(
        self, messages: Sequence[dict[str, Any]], max_new_tokens: int
    ) -> PipelineGeneration:
        wire_messages = [dict(message) for message in messages]
        if not wire_messages:
            raise APIRequestError("messages must not be empty")
        return self._generate_locked(
            lambda: self.tokenizer.apply_chat_template(
                wire_messages,
                tokenize=True,
                add_generation_prompt=True,
            ),
            max_new_tokens,
        )

    def close(self) -> None:
        # Shutdown waits for an in-flight request so worker sockets cannot be
        # closed halfway through a cache mutation/reset sequence.
        with self._request_lock:
            if self._closed:
                return
            self._closed = True
            self.pipeline.close()


def _validate_model(engine: Qwen35PipelineAPIEngine, requested_model: str) -> None:
    if requested_model != engine.model_id:
        raise APIRequestError(
            f"model {requested_model!r} is not served",
            status_code=HTTPStatus.NOT_FOUND,
        )


def _validate_greedy_request(
    request: CompletionRequest | ChatCompletionRequest,
) -> None:
    if request.stream:
        raise APIRequestError("streaming is not supported by this pipeline")
    if request.n != 1:
        raise APIRequestError("only n=1 is supported")
    if request.temperature not in (None, 0, 0.0):
        raise APIRequestError("only greedy temperature=0 is supported")
    if request.top_p not in (None, 1, 1.0):
        raise APIRequestError("only top_p=1 is supported")
    if request.stop not in (None, [], ""):
        raise APIRequestError("custom stop strings are not supported")
    if request.frequency_penalty not in (None, 0, 0.0):
        raise APIRequestError("frequency_penalty is not supported")
    if request.presence_penalty not in (None, 0, 0.0):
        raise APIRequestError("presence_penalty is not supported")
    if request.logit_bias:
        raise APIRequestError("logit_bias is not supported")
    if isinstance(request, CompletionRequest):
        if request.echo:
            raise APIRequestError("echo is not supported")
        if request.best_of not in (None, 1):
            raise APIRequestError("only best_of=1 is supported")
        if request.logprobs is not None:
            raise APIRequestError("logprobs are not supported")
        if request.suffix is not None:
            raise APIRequestError("suffix is not supported")
    else:
        if request.logprobs:
            raise APIRequestError("logprobs are not supported")
        if request.top_logprobs is not None:
            raise APIRequestError("top_logprobs are not supported")


def _chat_max_tokens(request: ChatCompletionRequest) -> int:
    if request.max_tokens is not None and request.max_completion_tokens is not None:
        raise APIRequestError("set only one of max_tokens and max_completion_tokens")
    return request.max_completion_tokens or request.max_tokens or 16


def _openai_usage(result: PipelineGeneration) -> dict[str, int]:
    prompt_tokens = len(result.prompt_token_ids)
    completion_tokens = len(result.completion_token_ids)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _token_metadata(result: PipelineGeneration) -> dict[str, list[int]]:
    return {
        "prompt_token_ids": list(result.prompt_token_ids),
        "completion_token_ids": list(result.completion_token_ids),
    }


def _error_response(exc: BaseException) -> JSONResponse:
    if isinstance(exc, PipelineBusyError):
        status = HTTPStatus.TOO_MANY_REQUESTS
        error_type = "server_busy"
    elif isinstance(exc, APIRequestError):
        status = HTTPStatus(exc.status_code)
        error_type = (
            "model_not_found"
            if status == HTTPStatus.NOT_FOUND
            else "invalid_request_error"
        )
    elif isinstance(exc, (ValueError, TypeError)):
        status = HTTPStatus.BAD_REQUEST
        error_type = "invalid_request_error"
    elif isinstance(exc, PipelineWorkerError):
        status = HTTPStatus.SERVICE_UNAVAILABLE
        error_type = "worker_error"
        logger.exception("Qwen3.5 pipeline worker request failed")
    elif isinstance(exc, PipelineProtocolError):
        status = HTTPStatus.INTERNAL_SERVER_ERROR
        error_type = "protocol_error"
        logger.exception("Qwen3.5 pipeline protocol failed")
    else:
        status = HTTPStatus.INTERNAL_SERVER_ERROR
        error_type = "server_error"
        logger.exception("Qwen3.5 pipeline API request failed")
    return JSONResponse(
        status_code=int(status),
        content={
            "error": {
                "message": str(exc),
                "type": error_type,
                "param": None,
                "code": int(status),
            }
        },
    )


def create_app(
    engine: Qwen35PipelineAPIEngine,
    *,
    api_key: str | None = None,
) -> FastAPI:
    """Create an app around an already-loaded engine, enabling CPU-only tests."""
    if api_key == "":
        raise ValueError("api_key must not be empty")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await run_in_threadpool(engine.close)

    app = FastAPI(title="Qwen3.5 MoE Dual-GPU Pipeline", lifespan=lifespan)
    app.state.pipeline_engine = engine

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, exc: RequestValidationError):
        return _error_response(APIRequestError(str(exc)))

    def authenticate(request: Request) -> JSONResponse | None:
        if api_key is None:
            return None
        supplied = request.headers.get("authorization", "")
        if hmac.compare_digest(supplied, f"Bearer {api_key}"):
            return None
        return JSONResponse(
            status_code=HTTPStatus.UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
            content={
                "error": {
                    "message": "invalid API key",
                    "type": "authentication_error",
                    "param": None,
                    "code": HTTPStatus.UNAUTHORIZED,
                }
            },
        )

    async def invoke(call: Callable[[], PipelineGeneration]):
        try:
            return await run_in_threadpool(call)
        except Exception as exc:
            return _error_response(exc)

    @app.get("/health")
    async def health():
        return {
            "status": "closed" if engine.closed else "ok",
            "busy": engine.busy,
            "model": engine.model_id,
        }

    @app.get("/v1/models")
    async def models(raw_request: Request):
        unauthorized = authenticate(raw_request)
        if unauthorized is not None:
            return unauthorized
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "sglang-qwen35-pipeline",
                }
            ],
        }

    @app.post("/generate")
    async def generate(request: NativeGenerateRequest, raw_request: Request):
        unauthorized = authenticate(raw_request)
        if unauthorized is not None:
            return unauthorized

        def run() -> PipelineGeneration:
            selected = sum(
                value is not None
                for value in (request.text, request.input_ids, request.messages)
            )
            if selected != 1:
                raise APIRequestError("set exactly one of text, input_ids, or messages")
            if request.messages is not None:
                if not request.messages:
                    raise APIRequestError("messages must not be empty")
                messages = [
                    message.model_dump(exclude_none=True)
                    for message in request.messages
                ]
                return engine.complete_chat(messages, request.max_new_tokens)
            prompt: str | Sequence[int]
            prompt = (
                request.text if request.text is not None else request.input_ids or []
            )
            return engine.complete_raw(prompt, request.max_new_tokens)

        result = await invoke(run)
        if isinstance(result, JSONResponse):
            return result
        return {
            "id": "gen-" + uuid.uuid4().hex,
            "text": result.text,
            "token_ids": list(result.completion_token_ids),
            "meta_info": {
                **_openai_usage(result),
                "finish_reason": result.finish_reason,
                **_token_metadata(result),
            },
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest, raw_request: Request):
        unauthorized = authenticate(raw_request)
        if unauthorized is not None:
            return unauthorized

        def run() -> PipelineGeneration:
            _validate_model(engine, request.model)
            _validate_greedy_request(request)
            return engine.complete_raw(request.prompt, request.max_tokens)

        result = await invoke(run)
        if isinstance(result, JSONResponse):
            return result
        return {
            "id": "cmpl-" + uuid.uuid4().hex,
            "object": "text_completion",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "text": result.text,
                    "logprobs": None,
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _openai_usage(result),
            "sglang": _token_metadata(result),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        unauthorized = authenticate(raw_request)
        if unauthorized is not None:
            return unauthorized

        def run() -> PipelineGeneration:
            _validate_model(engine, request.model)
            _validate_greedy_request(request)
            messages = [
                message.model_dump(exclude_none=True) for message in request.messages
            ]
            return engine.complete_chat(messages, _chat_max_tokens(request))

        result = await invoke(run)
        if isinstance(result, JSONResponse):
            return result
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "logprobs": None,
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _openai_usage(result),
            "sglang": _token_metadata(result),
        }

    return app


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    parser.add_argument("--served-model-name")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--v100-uuid", default=V100_UUID)
    parser.add_argument("--sm89-uuid", default=SM89_UUID)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY"))
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1,65535]")
    model_dir = Path(args.model_dir)
    model_id = args.served_model_name or model_dir.name
    tokenizer = load_tokenizer_compat(model_dir)
    pipeline = Qwen35Pipeline(
        model_dir,
        capacity=args.capacity,
        v100_uuid=args.v100_uuid,
        sm89_uuid=args.sm89_uuid,
    )
    engine = Qwen35PipelineAPIEngine(pipeline, tokenizer, model_id=model_id)
    app = create_app(engine, api_key=args.api_key)
    try:
        import uvicorn

        # One process is mandatory: the engine owns one pair of persistent GPU
        # workers and one cache slot.  Uvicorn must not fork additional copies.
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            workers=1,
        )
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
