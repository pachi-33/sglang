# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py
"""
Benchmark online serving.

Usage:
python3 -m sglang.bench_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
python3 -m sglang.bench_serving --backend sglang --dataset-name random --request-rate-range 1,2,4,8,16,32 --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --multi
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser as FlexibleArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, AsyncIterable, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: Union[str, List[int]]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    expert_trace: bool = False
    request_index: Optional[int] = None
    stream: Optional[bool] = None
    mock_expert_prefetch: Optional[dict] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    request_id: str = ""
    reported_prompt_len: Optional[int] = None
    completion_token_ids: List[int] = field(default_factory=list)
    record_request_id: str = ""
    record_latency: float = 0.0
    mock_expert_prefetch_metrics: Optional[dict] = None


def _prompt_sha256(prompt: Union[str, List[int]]) -> str:
    if isinstance(prompt, str):
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return hashlib.sha256(np.asarray(prompt, dtype="<i4").tobytes()).hexdigest()


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


async def _iter_sse_data(chunks: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
    """Yield complete SSE data payloads independently of transport chunking.

    HTTP clients may split one SSE record across chunks or coalesce multiple
    records into one chunk.  SSE frames are delimited by a blank line, and
    multiple ``data:`` lines in one frame are joined with a newline.
    """
    buffer = bytearray()
    data_lines: list[bytes] = []

    def consume_line(raw_line: bytes) -> str | None:
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        if not raw_line:
            if not data_lines:
                return None
            payload = b"\n".join(data_lines).decode("utf-8")
            data_lines.clear()
            return payload
        if raw_line.startswith(b":"):
            return None
        field, separator, value = raw_line.partition(b":")
        if field != b"data":
            return None
        if not separator:
            value = b""
        elif value.startswith(b" "):
            value = value[1:]
        data_lines.append(value)
        return None

    async for chunk in chunks:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("SSE transport chunks must be bytes-like")
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            raw_line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            payload = consume_line(raw_line)
            if payload is not None:
                yield payload

    if buffer:
        payload = consume_line(bytes(buffer))
        if payload is not None:
            yield payload
    if data_lines:
        yield b"\n".join(data_lines).decode("utf-8")


# trt llm not support ignore_eos
# https://github.com/triton-inference-server/tensorrtllm_backend/issues/505
async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.000001,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "min_length": request_func_input.output_len,
            "end_id": 1048576,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.output_len = request_func_input.output_len

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": (
                not args.disable_stream
                if request_func_input.stream is None
                else request_func_input.stream
            ),
            "ignore_eos": True,
        }
        if request_func_input.expert_trace:
            payload["expert_trace"] = True
        if request_func_input.mock_expert_prefetch is not None:
            payload["mock_expert_prefetch"] = request_func_input.mock_expert_prefetch
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        first_token_seen = False
        reported_completion_tokens = None
        reported_prompt_tokens = None
        st = time.perf_counter()
        most_recent_timestamp = st
        last_event_timestamp = st

        def record_data(data, timestamp, *, streaming):
            nonlocal first_token_seen, generated_text
            nonlocal most_recent_timestamp, reported_completion_tokens
            nonlocal reported_prompt_tokens
            if "error" in data:
                raise RuntimeError(
                    "completion API error: "
                    + json.dumps(data["error"], ensure_ascii=False)
                )

            event_request_id = data.get("id")
            if event_request_id is not None:
                if not isinstance(event_request_id, str) or not event_request_id:
                    raise RuntimeError("completion API returned an invalid response ID")
                if output.request_id and output.request_id != event_request_id:
                    raise RuntimeError(
                        "completion API changed response ID within one request: "
                        f"{output.request_id} != {event_request_id}"
                    )
                output.request_id = event_request_id

            usage = data.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                completion_tokens = usage["completion_tokens"]
                if (
                    isinstance(completion_tokens, bool)
                    or not isinstance(completion_tokens, int)
                    or completion_tokens < 0
                ):
                    raise RuntimeError(
                        "completion API returned invalid usage.completion_tokens"
                    )
                reported_completion_tokens = completion_tokens
            if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
                prompt_tokens = usage["prompt_tokens"]
                if (
                    isinstance(prompt_tokens, bool)
                    or not isinstance(prompt_tokens, int)
                    or prompt_tokens < 0
                ):
                    raise RuntimeError(
                        "completion API returned invalid usage.prompt_tokens"
                    )
                reported_prompt_tokens = prompt_tokens

            # A final usage-only frame can legally omit choices.  This server's
            # token frames carry one incremental token ID even when decoding that
            # token produces an empty text delta.  Other OpenAI-compatible
            # servers fall back to nonempty text as the arrival signal.
            choices = data.get("choices") or []
            choice = choices[0] if choices else {}
            text = choice.get("text", "") or ""
            is_terminal = streaming and choice.get("finish_reason") is not None
            sglang_metadata = data.get("sglang")
            if isinstance(sglang_metadata, dict) and isinstance(
                sglang_metadata.get("completion_token_ids"), list
            ):
                token_ids = sglang_metadata["completion_token_ids"]
                if not streaming or is_terminal:
                    output.completion_token_ids = [int(token) for token in token_ids]
                elif len(token_ids) == 1:
                    output.completion_token_ids.append(int(token_ids[0]))
            mock_metrics = data.get("mock_expert_prefetch_metrics")
            if mock_metrics is not None:
                if not isinstance(mock_metrics, dict):
                    raise RuntimeError("mock expert prefetch metrics must be an object")
                output.mock_expert_prefetch_metrics = mock_metrics
            if (
                not is_terminal
                and isinstance(sglang_metadata, dict)
                and "completion_token_ids" in sglang_metadata
            ):
                token_arrived = bool(sglang_metadata["completion_token_ids"])
            else:
                token_arrived = bool(text) and not is_terminal
            generated_text += text
            if not token_arrived:
                return

            if not first_token_seen:
                output.ttft = timestamp - st
                first_token_seen = True
            else:
                output.itl.append(timestamp - most_recent_timestamp)
            most_recent_timestamp = timestamp

        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    if payload["stream"]:
                        done_seen = False
                        async for event_data in _iter_sse_data(response.content):
                            timestamp = time.perf_counter()
                            last_event_timestamp = timestamp
                            if event_data == "[DONE]":
                                done_seen = True
                                break
                            record_data(
                                json.loads(event_data), timestamp, streaming=True
                            )
                        if not done_seen:
                            raise RuntimeError(
                                "streaming response ended before data: [DONE]"
                            )
                    else:
                        body = bytearray()
                        async for chunk in response.content:
                            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                                raise TypeError(
                                    "HTTP response chunks must be bytes-like"
                                )
                            body.extend(chunk)
                        timestamp = time.perf_counter()
                        last_event_timestamp = timestamp
                        record_data(json.loads(bytes(body)), timestamp, streaming=False)

                    if not first_token_seen:
                        raise RuntimeError(
                            "successful completion response contained no output token"
                        )
                    if (
                        reported_completion_tokens is not None
                        and reported_completion_tokens != request_func_input.output_len
                    ):
                        raise RuntimeError(
                            "completion token count differs from the fixed request: "
                            f"{reported_completion_tokens} != "
                            f"{request_func_input.output_len}"
                        )
                    if request_func_input.expert_trace:
                        if not output.request_id:
                            raise RuntimeError(
                                "traced completion response did not contain an ID"
                            )
                        if reported_prompt_tokens is None:
                            raise RuntimeError(
                                "traced completion response did not report prompt_tokens"
                            )
                        if reported_prompt_tokens != request_func_input.prompt_len:
                            raise RuntimeError(
                                "prompt token count differs from the fixed request: "
                                f"{reported_prompt_tokens} != "
                                f"{request_func_input.prompt_len}"
                            )

                    output.generated_text = generated_text
                    output.success = True
                    output.reported_prompt_len = reported_prompt_tokens
                    output.latency = last_event_timestamp - st
                    output.output_len = (
                        request_func_input.output_len
                        if reported_completion_tokens is None
                        else reported_completion_tokens
                    )
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_mock_expert_prefetch(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Run an unmeasured record followed by the measured streaming replay."""
    config = request_func_input.mock_expert_prefetch
    if not isinstance(config, dict):
        raise ValueError("mock expert prefetch request requires a configuration")
    pair_id = config.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("mock expert prefetch pair_id is required")
    record_input = RequestFuncInput(
        model=request_func_input.model,
        prompt=request_func_input.prompt,
        api_url=request_func_input.api_url,
        prompt_len=request_func_input.prompt_len,
        output_len=request_func_input.output_len,
        request_index=request_func_input.request_index,
        stream=False,
        mock_expert_prefetch={"phase": "record", "pair_id": pair_id},
    )
    record = await async_request_openai_completions(record_input)
    if not record.success:
        record.error = "mock record failed: " + record.error
        if pbar:
            pbar.update(1)
        return record
    replay_config = dict(config)
    replay_config["phase"] = "replay"
    replay_input = RequestFuncInput(
        model=request_func_input.model,
        prompt=request_func_input.prompt,
        api_url=request_func_input.api_url,
        prompt_len=request_func_input.prompt_len,
        output_len=request_func_input.output_len,
        request_index=request_func_input.request_index,
        stream=True,
        mock_expert_prefetch=replay_config,
    )
    replay = await async_request_openai_completions(replay_input)
    replay.record_request_id = record.request_id
    replay.record_latency = record.latency
    if replay.success and record.completion_token_ids != replay.completion_token_ids:
        replay.success = False
        replay.error = "mock record and replay completion token IDs differ"
    if replay.success and replay.mock_expert_prefetch_metrics is None:
        replay.success = False
        replay.error = "mock replay response omitted prefetch metrics"
    if pbar:
        pbar.update(1)
    return replay


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("SGLANG_USE_MODELSCOPE", "False").lower() == "true":
        import huggingface_hub.constants
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
        )

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    if pretrained_model_name_or_path is not None and os.path.isdir(
        pretrained_model_name_or_path
    ):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        try:
            with open(config_path, encoding="utf-8") as file:
                model_type = json.load(file).get("model_type")
        except (OSError, json.JSONDecodeError, AttributeError):
            model_type = None
        if model_type == "qwen3_5_moe":
            # This checkpoint stores BPE merges as token pairs.  The
            # transformers/tokenizers versions pinned by the V100 branch need
            # the same in-memory compatibility conversion as the model API.
            from sglang.srt.layers.qwen3_5.pipeline import load_tokenizer_compat

            return load_tokenizer_compat(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_openai_completions,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "trt": async_request_trt_llm,
}


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


async def request_with_concurrency_limit(
    request_func,
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm],
    semaphore: Optional[asyncio.Semaphore],
) -> RequestFuncOutput:
    if semaphore is None:
        return await request_func(request_func_input=request_func_input, pbar=pbar)
    async with semaphore:
        return await request_func(request_func_input=request_func_input, pbar=pbar)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float


default_sharegpt_path = "ShareGPT_V3_unfiltered_cleaned_split.json"


def download_sharegpt_dataset(path):
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

    print(f"Downloading dataset from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192

        with open(path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                progress_bar.update(size)

        print(f"Dataset downloaded and saved to {path}")
    except requests.RequestException as e:
        raise Exception(f"Failed to download dataset: {e}")


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path) and not os.path.isfile(default_sharegpt_path):
        download_sharegpt_dataset(default_sharegpt_path)
        dataset_path = default_sharegpt_path
    else:
        dataset_path = (
            dataset_path if os.path.isfile(dataset_path) else default_sharegpt_path
        )

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    return_token_ids: bool = False,
) -> List[Tuple[Union[str, List[int]], int, int]]:

    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    if True:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        # Download sharegpt if necessary
        if not os.path.isfile(dataset_path) and not os.path.isfile(
            default_sharegpt_path
        ):
            download_sharegpt_dataset(default_sharegpt_path)
            dataset_path = default_sharegpt_path
        else:
            dataset_path = (
                dataset_path if os.path.isfile(dataset_path) else default_sharegpt_path
            )

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]

        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[Tuple[Union[str, List[int]], int, int]] = []
        for i in range(num_prompts):
            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            prompt_token_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_token_ids)

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            request_prompt = (
                input_ids if return_token_ids else tokenizer.decode(input_ids)
            )
            input_requests.append(
                (request_prompt, int(input_lens[i]), int(output_lens[i]))
            )
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            prompt = tokenizer.decode(
                [
                    (offsets[i] + i + j) % tokenizer.vocab_size
                    for j in range(input_lens[i])
                ]
            )
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests


async def get_request(
    input_requests: List[Tuple[Union[str, List[int]], int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[Union[str, List[int]], int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[Union[str, List[int]], int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    backend: str,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
    )

    return metrics, output_lens


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[Union[str, List[int]], int, int]],
    request_rate: float,
    disable_tqdm: bool,
    enable_multi: bool,
    max_concurrency: Optional[int] = None,
    expert_trace: bool = False,
    mock_expert_prefetch: Optional[dict] = None,
):
    if max_concurrency is not None and (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency <= 0
    ):
        raise ValueError("max_concurrency must be a positive integer or None")

    if backend in ASYNC_REQUEST_FUNCS:
        base_request_func = ASYNC_REQUEST_FUNCS[backend]
        request_func = (
            async_request_mock_expert_prefetch
            if mock_expert_prefetch is not None
            else base_request_func
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len = input_requests[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        expert_trace=False,
    )
    test_output = await base_request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    semaphore = (
        asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
    )
    request_index = 0
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        pair_prompt_digest = _prompt_sha256(prompt)
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            expert_trace=expert_trace,
            request_index=request_index,
            mock_expert_prefetch=(
                None
                if mock_expert_prefetch is None
                else {
                    **mock_expert_prefetch,
                    "pair_id": (
                        f"bench-s{mock_expert_prefetch['seed']}-"
                        f"{request_index}-{pair_prompt_digest[:16]}"
                    ),
                }
            ),
        )
        request_index += 1
        tasks.append(
            asyncio.create_task(
                request_with_concurrency_limit(
                    request_func,
                    request_func_input,
                    pbar,
                    semaphore,
                )
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    wall_duration = time.perf_counter() - benchmark_start_time
    benchmark_duration = (
        sum(output.latency for output in outputs if output.success)
        if mock_expert_prefetch is not None
        else wall_duration
    )

    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    if mock_expert_prefetch is not None:
        print(
            "{:<40} {:<10.2f}".format(
                "Full Record+Replay wall time (s):", wall_duration
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Excluded Record duration (s):",
                sum(output.record_latency for output in outputs),
            )
        )
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            "backend": args.backend,
            "dataset_name": args.dataset_name,
            "request_rate": request_rate,
            "completed": metrics.completed,
            "total_input": metrics.total_input,
            "total_output": metrics.total_output,
            "total_output_retokenized": metrics.total_output_retokenized,
            "mean_e2e_latency": metrics.mean_e2e_latency_ms,
            "median_e2e_latency": metrics.median_e2e_latency_ms,
            "median_ttft": metrics.median_ttft_ms,
            "median_itl": metrics.median_itl_ms,
            "output_token_throughput": metrics.output_throughput,
            "sharegpt_output_len": args.sharegpt_output_len,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "benchmark_duration": benchmark_duration,
            "wall_duration": wall_duration,
        }
        if expert_trace:
            trace_requests = []
            for index, ((prompt, prompt_len, output_len), output) in enumerate(
                zip(input_requests, outputs)
            ):
                if isinstance(prompt, str):
                    prompt_kind = "text"
                    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                else:
                    prompt_kind = "token_ids_int32_le"
                    prompt_sha256 = hashlib.sha256(
                        np.asarray(prompt, dtype="<i4").tobytes()
                    ).hexdigest()
                trace_requests.append(
                    {
                        "request_index": index,
                        "trace_id": output.request_id or None,
                        "success": output.success,
                        "prompt_kind": prompt_kind,
                        "prompt_sha256": prompt_sha256,
                        "prompt_tokens": prompt_len,
                        "reported_prompt_tokens": output.reported_prompt_len,
                        "completion_tokens": output_len,
                        "error": output.error,
                    }
                )
            result["expert_trace"] = True
            result["trace_requests"] = trace_requests
        if mock_expert_prefetch is not None:
            result["mock_expert_prefetch"] = dict(mock_expert_prefetch)
            result["record_duration"] = sum(output.record_latency for output in outputs)
            result["mock_requests"] = [
                {
                    "request_index": index,
                    "record_response_id": output.record_request_id or None,
                    "replay_response_id": output.request_id or None,
                    "success": output.success,
                    "record_duration": output.record_latency,
                    "replay_duration": output.latency,
                    "tokens_equal": output.success,
                    "prompt_sha256": _prompt_sha256(input_requests[index][0]),
                    "prompt_tokens": input_requests[index][1],
                    "completion_tokens": input_requests[index][2],
                    "prefetch_metrics": output.mock_expert_prefetch_metrics,
                    "error": output.error,
                }
                for index, output in enumerate(outputs)
            ]
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        if args.dataset_name == "random":
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
        else:
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_sharegpt.jsonl"

    # Append results to a JSONL file
    with open(output_file_name, "a") as file:
        file.write(json.dumps(result) + "\n")

    result = {
        "duration": benchmark_duration,
        "wall_duration": wall_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
    }
    if mock_expert_prefetch is not None:
        result["record_duration"] = sum(output.record_latency for output in outputs)
        result["mock_expert_prefetch_metrics"] = [
            output.mock_expert_prefetch_metrics for output in outputs
        ]
    return result


def parse_request_rate_range(request_rate_range):
    if len(request_rate_range.split(",")) == 3:
        start, stop, step = map(int, request_rate_range.split(","))
        return list(range(start, stop, step))
    else:
        return list(map(int, request_rate_range.split(",")))


def check_chat_template(model_path):
    try:
        tokenizer = get_tokenizer(model_path)
        return bool(getattr(tokenizer, "chat_template", None))
    except Exception as e:
        print(f"Fail to load tokenizer config with error={e}")
        return False


def fire(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.expert_trace and args.backend != "sglang":
        raise ValueError("--expert-trace is only supported by the sglang backend")
    if args.random_input_token_ids and args.dataset_name != "random":
        raise ValueError("--random-input-token-ids requires --dataset-name random")
    if args.mock_expert_prefetch:
        if args.backend != "sglang":
            raise ValueError("--mock-expert-prefetch requires --backend sglang")
        if args.expert_trace:
            raise ValueError(
                "mock expert prefetch and expert trace are mutually exclusive"
            )
        if args.disable_stream:
            raise ValueError("mock expert prefetch requires streaming Replay")
        if args.max_concurrency != 1:
            raise ValueError("mock expert prefetch requires --max-concurrency 1")
        if not math.isinf(args.request_rate):
            raise ValueError("mock expert prefetch requires --request-rate inf")
        maximum_recall = min(1.0, args.mock_prefetch_top_k / 8)
        if not 0 <= args.mock_prefetch_recall <= maximum_recall:
            raise ValueError(
                "--mock-prefetch-recall exceeds the Top-8 recall possible "
                "for the configured top-k"
            )

    if args.port is None:
        args.port = {
            "sglang": 30000,
            "lmdeploy": 23333,
            "vllm": 8000,
            "trt": 8000,
        }.get(args.backend, 30000)

    api_url = (
        f"{args.base_url}/v1/completions"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/completions"
    )
    model_url = (
        f"{args.base_url}/v1/models"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/models"
    )

    if args.backend == "trt":
        api_url = (
            f"{args.base_url}/v2/models/ensemble/generate_stream"
            if args.base_url
            else f"http://{args.host}:{args.port}/v2/models/ensemble/generate_stream"
        )
        if args.model is None:
            print("Please provide a model using `--model` when using `trt` backend.")
            sys.exit(1)

    if args.model is None:
        try:
            response = requests.get(model_url)
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            print(
                "Please specify the correct host and port using `--host` and `--port`."
            )
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    if not check_chat_template(args.model):
        print(
            "\nWARNING It is recommended to use the `Chat` or `Instruct` model for benchmarking.\n"
            "Because when the tokenizer counts the output tokens, if there is gibberish, it might count incorrectly.\n"
        )

    print(f"{args}\n")

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    tokenizer = get_tokenizer(tokenizer_id)

    if args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )
    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            return_token_ids=args.random_input_token_ids,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    if args.multi:
        request_rates = parse_request_rate_range(args.request_rate_range)

        for rate in request_rates:
            asyncio.run(
                benchmark(
                    backend=backend,
                    api_url=api_url,
                    model_id=model_id,
                    tokenizer=tokenizer,
                    input_requests=input_requests,
                    request_rate=rate,
                    disable_tqdm=args.disable_tqdm,
                    enable_multi=args.multi,
                    max_concurrency=args.max_concurrency,
                    expert_trace=args.expert_trace,
                    mock_expert_prefetch=(
                        None
                        if not args.mock_expert_prefetch
                        else {
                            "route_recall": args.mock_prefetch_recall,
                            "top_k": args.mock_prefetch_top_k,
                            "lead_layers": args.mock_prefetch_lead_layers,
                            "seed": args.mock_prefetch_seed,
                        }
                    ),
                )
            )
    else:
        asyncio.run(
            benchmark(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                tokenizer=tokenizer,
                input_requests=input_requests,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm,
                enable_multi=args.multi,
                max_concurrency=args.max_concurrency,
                expert_trace=args.expert_trace,
                mock_expert_prefetch=(
                    None
                    if not args.mock_expert_prefetch
                    else {
                        "route_recall": args.mock_prefetch_recall,
                        "top_k": args.mock_prefetch_top_k,
                        "lead_layers": args.mock_prefetch_lead_layers,
                        "seed": args.mock_prefetch_seed,
                    }
                ),
            )
        )


# to avoid relying on SGLang's components
def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "random"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is 128.0.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=positive_int,
        default=None,
        help="Maximum number of in-flight requests. By default, concurrency is unlimited.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Default is 0.")
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use request rate range rather than single value.",
    )
    parser.add_argument(
        "--request-rate-range",
        type=str,
        default="2,34,2",
        help="Range of request rates in the format start,stop,step. Default is 2,34,2. It also supports a list of request rates, requiring the parameters to not equal three.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--expert-trace",
        action="store_true",
        help=(
            "Request server-side expert traces for benchmark requests. The "
            "initial validation request remains untraced."
        ),
    )
    parser.add_argument(
        "--random-input-token-ids",
        action="store_true",
        help=(
            "For the random dataset, send the sampled token IDs directly "
            "instead of decoding them back to text."
        ),
    )
    parser.add_argument(
        "--mock-expert-prefetch",
        action="store_true",
        help="Run an unmeasured route-record pass before each measured Replay.",
    )
    parser.add_argument("--mock-prefetch-recall", type=float, default=0.5)
    parser.add_argument("--mock-prefetch-top-k", type=int, default=8)
    parser.add_argument("--mock-prefetch-lead-layers", type=int, default=2)
    parser.add_argument("--mock-prefetch-seed", type=int, default=0)

    set_ulimit()

    args = parser.parse_args()
    fire(args)
