"""HTTP API for a narrow Qwen3.5 single-request generation backend.

This adapter deliberately preserves the pipeline's narrow execution contract:
one active request, batch one, and greedy decoding.  It exposes a small native
``/generate`` endpoint plus OpenAI-style completion and chat-completion
endpoints.  Backends that implement ``generate_ids_stream`` can stream sampled
tokens; the older pipeline backend remains non-streaming.  This adapter does
not route through SGLang's normal scheduler, radix cache, or general server
state.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Protocol, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from starlette.concurrency import run_in_threadpool

from .pipeline import (
    DEFAULT_BACK_UUID,
    DEFAULT_FRONT_UUID,
    DEFAULT_SPLIT_LAYER,
    EOS_TOKEN_IDS,
    MODEL_DIR_DEFAULT,
    PipelineProtocolError,
    PipelineWorkerError,
    Qwen35Pipeline,
    load_tokenizer_compat,
)

logger = logging.getLogger(__name__)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
StrictNumber = StrictInt | StrictFloat


class Qwen35GenerationBackend(Protocol):
    """The common surface implemented by pipeline and single-GPU backends."""

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
    ) -> list[int]: ...

    def close(self) -> None: ...


class Qwen35StreamingGenerationBackend(Qwen35GenerationBackend, Protocol):
    """Optional extension implemented by the single-GPU backend."""

    def generate_ids_stream(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        token_callback: Callable[[int], None],
    ) -> list[int]: ...


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
    # SGLang's serving benchmark sets this extension to force a fixed output
    # length.  The backend remains greedy; an empty EOS set only disables the
    # early-stop check.
    ignore_eos: StrictBool = False
    expert_trace: StrictBool = False


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
    ignore_eos: StrictBool = False
    expert_trace: StrictBool = False


class NativeGenerateRequest(_RequestModel):
    text: str | None = None
    input_ids: list[StrictInt] | None = None
    messages: list[ChatMessageRequest] | None = None
    max_new_tokens: StrictInt = Field(default=16, ge=1, le=2048)
    expert_trace: StrictBool = False


@dataclass(frozen=True)
class PipelineGeneration:
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    text: str
    finish_reason: Literal["stop", "length"]


@dataclass(frozen=True)
class _StreamToken:
    token_id: int
    text: str


@dataclass(frozen=True)
class _StreamTerminal:
    result: PipelineGeneration | None = None
    error: BaseException | None = None
    text: str = ""


class _PipelineGenerationStream:
    """Own a backend generation thread and its already-acquired request slot.

    The worker, rather than the response iterator, releases the engine lock.
    Consequently an HTTP disconnect cannot make the transactional request cache
    reusable while generation or backend cleanup is still in progress.
    """

    def __init__(
        self,
        engine: "Qwen35PipelineAPIEngine",
        generate: Callable[..., list[int]],
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.prompt_ids = tuple(prompt_ids)
        self.max_new_tokens = max_new_tokens
        self.eos_token_ids = tuple(eos_token_ids)
        self.generation_kwargs = dict(generation_kwargs or {})
        self._generate = generate
        self._events: queue.Queue[_StreamToken | _StreamTerminal] = queue.Queue()
        self._prefetched: _StreamToken | _StreamTerminal | None = None
        self._cancelled = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="qwen35-api-generation",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def prefetch(self) -> _StreamToken | _StreamTerminal:
        if self._prefetched is not None:
            raise RuntimeError("stream already prefetched")
        self._prefetched = self._events.get()
        return self._prefetched

    def cancel(self) -> None:
        self._cancelled.set()

    def __iter__(self) -> Iterator[_StreamToken | _StreamTerminal]:
        if self._prefetched is not None:
            event = self._prefetched
            self._prefetched = None
            yield event
            if isinstance(event, _StreamTerminal):
                return
        while True:
            event = self._events.get()
            yield event
            if isinstance(event, _StreamTerminal):
                return

    def _run(self) -> None:
        callback_ids: list[int] = []
        visible_ids: list[int] = []
        committed_text = ""
        eos = frozenset(self.eos_token_ids)
        terminal: _StreamTerminal

        def token_callback(token_id: int) -> None:
            nonlocal committed_text
            if self._cancelled.is_set():
                raise RuntimeError("streaming client disconnected")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise PipelineProtocolError(
                    "generate_ids_stream token callback must receive Python ints"
                )
            callback_ids.append(token_id)
            if token_id in eos:
                return
            visible_ids.append(token_id)
            current_text = self.engine.tokenizer.decode(
                visible_ids, skip_special_tokens=False
            )
            if not current_text.startswith(committed_text):
                raise PipelineProtocolError(
                    "incremental detokenization rewrote already-streamed text"
                )
            delta = current_text[len(committed_text) :]
            is_final_length_token = len(callback_ids) >= self.max_new_tokens
            if not is_final_length_token and delta.endswith("\ufffd"):
                # Byte-level tokenizers can expose a temporary replacement
                # character until later tokens complete the UTF-8 sequence.
                # Keep the token event (and its timing metadata), but hold the
                # unstable text so clients never receive text that must later
                # be rewritten.
                delta = ""
            else:
                committed_text = current_text
            self._events.put(_StreamToken(token_id=token_id, text=delta))

        try:
            generated = list(
                self._generate(
                    self.prompt_ids,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_ids=self.eos_token_ids,
                    token_callback=token_callback,
                    **self.generation_kwargs,
                )
            )
            if generated != callback_ids:
                raise PipelineProtocolError(
                    "generate_ids_stream returned tokens that differ from its callbacks"
                )
            stopped = bool(generated and generated[-1] in eos)
            result = PipelineGeneration(
                prompt_token_ids=self.prompt_ids,
                completion_token_ids=tuple(generated),
                text=self.engine.tokenizer.decode(
                    [token for token in generated if token not in eos],
                    skip_special_tokens=False,
                ),
                finish_reason="stop" if stopped else "length",
            )
            if not result.text.startswith(committed_text):
                raise PipelineProtocolError(
                    "final detokenization rewrote already-streamed text"
                )
            terminal = _StreamTerminal(
                result=result,
                text=result.text[len(committed_text) :],
            )
        except BaseException as exc:
            terminal = _StreamTerminal(error=exc)
        finally:
            # ``generate_ids_stream`` returns or raises only after its own
            # request-cache reset.  Publish the terminal event after releasing
            # the slot so DONE/error means cleanup has completed.
            self.engine._request_lock.release()
        self._events.put(terminal)


class _PipelineStreamingResponse(StreamingResponse):
    """Signal disconnects without releasing the backend request slot."""

    def __init__(self, stream: _PipelineGenerationStream, content: Any, **kwargs):
        self._generation_stream = stream
        super().__init__(content, **kwargs)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # The worker observes this at a token boundary, lets the backend
            # perform its transactional reset, and only then releases the lock.
            self._generation_stream.cancel()


class Qwen35PipelineAPIEngine:
    """Serialize HTTP calls onto a backend's one reusable request slot."""

    def __init__(
        self,
        pipeline: Qwen35GenerationBackend,
        tokenizer: Any,
        *,
        model_id: str,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        # Keep ``pipeline`` as a compatibility alias for existing callers and
        # tests while allowing the single-GPU implementation to share the API.
        self.backend = pipeline
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

    @property
    def failed(self) -> bool:
        value = getattr(self.backend, "failed", False)
        return bool(value() if callable(value) else value)

    @property
    def stats(self) -> dict[str, Any]:
        value = getattr(self.backend, "stats", None)
        if callable(value):
            value = value()
        return {} if value is None else dict(value)

    def _generate_locked(
        self,
        prompt_builder: Callable[[], Sequence[int]],
        max_new_tokens: int,
        *,
        ignore_eos: bool = False,
        expert_trace: bool = False,
        request_id: str | None = None,
    ) -> PipelineGeneration:
        if not self._request_lock.acquire(blocking=False):
            raise PipelineBusyError("the single pipeline request slot is busy")
        try:
            if self._closed:
                raise RuntimeError("the pipeline API engine is closed")
            if self.failed:
                raise RuntimeError(
                    "the Qwen3.5 generation backend has failed; restart the process"
                )
            prompt_ids = list(prompt_builder())
            if not prompt_ids:
                raise APIRequestError("prompt must produce at least one token")
            eos_token_ids = () if ignore_eos else EOS_TOKEN_IDS
            trace_kwargs = self._expert_trace_kwargs(expert_trace, request_id)
            generated = self.pipeline.generate_ids(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                **trace_kwargs,
            )
            stopped = bool(generated and generated[-1] in eos_token_ids)
            visible_ids = [token for token in generated if token not in eos_token_ids]
            return PipelineGeneration(
                prompt_token_ids=tuple(prompt_ids),
                completion_token_ids=tuple(generated),
                text=self.tokenizer.decode(visible_ids, skip_special_tokens=False),
                finish_reason="stop" if stopped else "length",
            )
        finally:
            self._request_lock.release()

    def _begin_stream_locked(
        self,
        prompt_builder: Callable[[], Sequence[int]],
        max_new_tokens: int,
        *,
        ignore_eos: bool = False,
        expert_trace: bool = False,
        request_id: str | None = None,
    ) -> _PipelineGenerationStream:
        generate = getattr(self.backend, "generate_ids_stream", None)
        if not callable(generate):
            raise APIRequestError(
                "streaming is not supported by this generation backend"
            )
        if not self._request_lock.acquire(blocking=False):
            raise PipelineBusyError("the single pipeline request slot is busy")
        ownership_transferred = False
        try:
            if self._closed:
                raise RuntimeError("the pipeline API engine is closed")
            if self.failed:
                raise RuntimeError(
                    "the Qwen3.5 generation backend has failed; restart the process"
                )
            prompt_ids = list(prompt_builder())
            if not prompt_ids:
                raise APIRequestError("prompt must produce at least one token")
            eos_token_ids = () if ignore_eos else EOS_TOKEN_IDS
            trace_kwargs = self._expert_trace_kwargs(expert_trace, request_id)
            stream = _PipelineGenerationStream(
                self,
                generate,
                prompt_ids,
                max_new_tokens,
                eos_token_ids,
                trace_kwargs,
            )
            stream.start()
            ownership_transferred = True
            return stream
        finally:
            if not ownership_transferred:
                self._request_lock.release()

    def complete_raw(
        self,
        prompt: str | Sequence[int],
        max_new_tokens: int,
        *,
        ignore_eos: bool = False,
        expert_trace: bool = False,
        request_id: str | None = None,
    ) -> PipelineGeneration:
        if isinstance(prompt, str):
            if not prompt:
                raise APIRequestError("prompt must not be empty")
            builder = lambda: self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_ids = list(prompt)
            builder = lambda: prompt_ids
        return self._generate_locked(
            builder,
            max_new_tokens,
            ignore_eos=ignore_eos,
            expert_trace=expert_trace,
            request_id=request_id,
        )

    def stream_raw(
        self,
        prompt: str | Sequence[int],
        max_new_tokens: int,
        *,
        ignore_eos: bool = False,
        expert_trace: bool = False,
        request_id: str | None = None,
    ) -> _PipelineGenerationStream:
        if isinstance(prompt, str):
            if not prompt:
                raise APIRequestError("prompt must not be empty")
            builder = lambda: self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_ids = list(prompt)
            builder = lambda: prompt_ids
        return self._begin_stream_locked(
            builder,
            max_new_tokens,
            ignore_eos=ignore_eos,
            expert_trace=expert_trace,
            request_id=request_id,
        )

    def complete_chat(
        self,
        messages: Sequence[dict[str, Any]],
        max_new_tokens: int,
        *,
        ignore_eos: bool = False,
        expert_trace: bool = False,
        request_id: str | None = None,
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
            ignore_eos=ignore_eos,
            expert_trace=expert_trace,
            request_id=request_id,
        )

    def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        max_new_tokens: int,
        *,
        ignore_eos: bool = False,
        expert_trace: bool = False,
        request_id: str | None = None,
    ) -> _PipelineGenerationStream:
        wire_messages = [dict(message) for message in messages]
        if not wire_messages:
            raise APIRequestError("messages must not be empty")
        return self._begin_stream_locked(
            lambda: self.tokenizer.apply_chat_template(
                wire_messages,
                tokenize=True,
                add_generation_prompt=True,
            ),
            max_new_tokens,
            ignore_eos=ignore_eos,
            expert_trace=expert_trace,
            request_id=request_id,
        )

    def _expert_trace_kwargs(
        self, expert_trace: bool, request_id: str | None
    ) -> dict[str, Any]:
        if not expert_trace:
            return {}
        enabled = getattr(self.backend, "expert_trace_enabled", False)
        if callable(enabled):
            enabled = enabled()
        if not enabled:
            raise APIRequestError(
                "expert trace was requested but this backend has no trace directory"
            )
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("an expert trace request requires a server request ID")
        return {"expert_trace": True, "request_id": request_id}

    def close(self) -> None:
        # Shutdown waits for an in-flight request so worker sockets cannot be
        # closed halfway through a cache mutation/reset sequence.
        with self._request_lock:
            if self._closed:
                return
            self._closed = True
            self.backend.close()


def _validate_model(engine: Qwen35PipelineAPIEngine, requested_model: str) -> None:
    if requested_model != engine.model_id:
        raise APIRequestError(
            f"model {requested_model!r} is not served",
            status_code=HTTPStatus.NOT_FOUND,
        )


def _validate_greedy_request(
    request: CompletionRequest | ChatCompletionRequest,
) -> None:
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


def _error_response(
    exc: BaseException, *, backend_failed: bool = False
) -> JSONResponse:
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
    elif backend_failed:
        status = HTTPStatus.SERVICE_UNAVAILABLE
        error_type = "backend_failed"
        logger.exception("Qwen3.5 generation backend has failed")
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


def _sse_json(content: dict[str, Any]) -> str:
    return f"data: {json.dumps(content, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _stream_error_json(exc: BaseException, *, backend_failed: bool) -> dict[str, Any]:
    response = _error_response(exc, backend_failed=backend_failed)
    return json.loads(response.body)


def _completion_sse(
    stream: _PipelineGenerationStream,
    *,
    request_id: str,
    created: int,
    model: str,
) -> Iterator[str]:
    for event in stream:
        if isinstance(event, _StreamToken):
            yield _sse_json(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": event.text,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                    "usage": None,
                    "sglang": {"completion_token_ids": [event.token_id]},
                }
            )
            continue

        if event.error is not None:
            yield _sse_json(
                _stream_error_json(event.error, backend_failed=stream.engine.failed)
            )
        else:
            assert event.result is not None
            yield _sse_json(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": event.text,
                            "logprobs": None,
                            "finish_reason": event.result.finish_reason,
                        }
                    ],
                    "usage": _openai_usage(event.result),
                    "sglang": _token_metadata(event.result),
                }
            )
        yield "data: [DONE]\n\n"


def _chat_completion_sse(
    stream: _PipelineGenerationStream,
    *,
    request_id: str,
    created: int,
    model: str,
) -> Iterator[str]:
    # Match the OpenAI chat protocol's initial assistant-role delta.
    yield _sse_json(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "usage": None,
        }
    )
    for event in stream:
        if isinstance(event, _StreamToken):
            yield _sse_json(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": event.text},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                    "usage": None,
                    "sglang": {"completion_token_ids": [event.token_id]},
                }
            )
            continue

        if event.error is not None:
            yield _sse_json(
                _stream_error_json(event.error, backend_failed=stream.engine.failed)
            )
        else:
            assert event.result is not None
            yield _sse_json(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": ({"content": event.text} if event.text else {}),
                            "logprobs": None,
                            "finish_reason": event.result.finish_reason,
                        }
                    ],
                    "usage": _openai_usage(event.result),
                    "sglang": _token_metadata(event.result),
                }
            )
        yield "data: [DONE]\n\n"


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
    app.state.generation_engine = engine

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

    async def invoke(call: Callable[[], Any]):
        try:
            if engine.failed:
                raise RuntimeError(
                    "the Qwen3.5 generation backend has failed; restart the process"
                )
            return await run_in_threadpool(call)
        except Exception as exc:
            return _error_response(exc, backend_failed=engine.failed)

    async def stream_handshake(
        stream: _PipelineGenerationStream,
    ) -> JSONResponse | None:
        first_event = await run_in_threadpool(stream.prefetch)
        if isinstance(first_event, _StreamTerminal) and first_event.error is not None:
            return _error_response(first_event.error, backend_failed=engine.failed)
        return None

    @app.get("/health")
    async def health():
        status = "failed" if engine.failed else "closed" if engine.closed else "ok"
        content = {
            "status": status,
            "busy": engine.busy,
            "model": engine.model_id,
        }
        try:
            stats = engine.stats
        except Exception as exc:
            if not engine.failed:
                raise
            # Health must remain a reliable restart signal even if failure
            # diagnostics themselves can no longer be snapshotted.
            stats = {}
            content["stats_error"] = f"{type(exc).__name__}: {exc}"
        if stats:
            content["stats"] = stats
        if engine.failed:
            return JSONResponse(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE, content=content
            )
        return content

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
        request_id = "gen-" + uuid.uuid4().hex

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
                return engine.complete_chat(
                    messages,
                    request.max_new_tokens,
                    expert_trace=request.expert_trace,
                    request_id=request_id,
                )
            prompt: str | Sequence[int]
            prompt = (
                request.text if request.text is not None else request.input_ids or []
            )
            return engine.complete_raw(
                prompt,
                request.max_new_tokens,
                expert_trace=request.expert_trace,
                request_id=request_id,
            )

        result = await invoke(run)
        if isinstance(result, JSONResponse):
            return result
        return {
            "id": request_id,
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
        request_id = "cmpl-" + uuid.uuid4().hex

        def run() -> PipelineGeneration | _PipelineGenerationStream:
            _validate_model(engine, request.model)
            _validate_greedy_request(request)
            if request.stream:
                return engine.stream_raw(
                    request.prompt,
                    request.max_tokens,
                    ignore_eos=request.ignore_eos,
                    expert_trace=request.expert_trace,
                    request_id=request_id,
                )
            return engine.complete_raw(
                request.prompt,
                request.max_tokens,
                ignore_eos=request.ignore_eos,
                expert_trace=request.expert_trace,
                request_id=request_id,
            )

        result = await invoke(run)
        if isinstance(result, JSONResponse):
            return result
        if isinstance(result, _PipelineGenerationStream):
            handshake_error = await stream_handshake(result)
            if handshake_error is not None:
                return handshake_error
            created = int(time.time())
            return _PipelineStreamingResponse(
                result,
                _completion_sse(
                    result,
                    request_id=request_id,
                    created=created,
                    model=engine.model_id,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return {
            "id": request_id,
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
        request_id = "chatcmpl-" + uuid.uuid4().hex

        def run() -> PipelineGeneration | _PipelineGenerationStream:
            _validate_model(engine, request.model)
            _validate_greedy_request(request)
            messages = [
                message.model_dump(exclude_none=True) for message in request.messages
            ]
            if request.stream:
                return engine.stream_chat(
                    messages,
                    _chat_max_tokens(request),
                    ignore_eos=request.ignore_eos,
                    expert_trace=request.expert_trace,
                    request_id=request_id,
                )
            return engine.complete_chat(
                messages,
                _chat_max_tokens(request),
                ignore_eos=request.ignore_eos,
                expert_trace=request.expert_trace,
                request_id=request_id,
            )

        result = await invoke(run)
        if isinstance(result, JSONResponse):
            return result
        if isinstance(result, _PipelineGenerationStream):
            handshake_error = await stream_handshake(result)
            if handshake_error is not None:
                return handshake_error
            created = int(time.time())
            return _PipelineStreamingResponse(
                result,
                _chat_completion_sse(
                    result,
                    request_id=request_id,
                    created=created,
                    model=engine.model_id,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return {
            "id": request_id,
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
    parser.add_argument(
        "--front-uuid",
        help=f"GPU UUID for embedding/front layers/head (default: {DEFAULT_FRONT_UUID})",
    )
    parser.add_argument(
        "--back-uuid",
        help=f"GPU UUID for remaining layers (default: {DEFAULT_BACK_UUID})",
    )
    parser.add_argument("--split-layer", type=int, default=DEFAULT_SPLIT_LAYER)
    parser.add_argument("--v100-uuid", help=argparse.SUPPRESS)
    parser.add_argument("--sm89-uuid", help=argparse.SUPPRESS)
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
        front_uuid=args.front_uuid,
        back_uuid=args.back_uuid,
        split_layer=args.split_layer,
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
