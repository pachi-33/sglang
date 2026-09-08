#!/usr/bin/env python3
"""Reproduce the auxiliary cold/warm request used by EXP-0001."""

import asyncio
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sglang import bench_serving

MODEL = "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16"
DATASET = "/home/yaozhenyang/dev/sglang-v100/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTPUT = Path(__file__).with_name("results") / "cold_warm_characterization.json"
BENCH_OUTPUT = Path(__file__).with_name("results") / "bench.jsonl"


async def main():
    random.seed(0)
    np.random.seed(0)
    tokenizer = bench_serving.get_tokenizer(MODEL)
    prompt, prompt_len, output_len = bench_serving.sample_random_requests(
        1024,
        256,
        128,
        1.0,
        tokenizer,
        DATASET,
        return_token_ids=True,
    )[0]
    prompt_sha256 = hashlib.sha256(
        np.asarray(prompt, dtype="<i4").tobytes()
    ).hexdigest()
    bench_record = json.loads(BENCH_OUTPUT.read_text(encoding="utf-8"))
    if prompt_sha256 != bench_record["trace_requests"][0]["prompt_sha256"]:
        raise RuntimeError("regenerated prompt does not match EXP-0001 request 0")
    bench_serving.args = SimpleNamespace(disable_stream=False)
    measurements = []
    for profile in ("cold", "warm"):
        result = await bench_serving.async_request_openai_completions(
            bench_serving.RequestFuncInput(
                prompt,
                "http://127.0.0.1:8818/v1/completions",
                prompt_len,
                output_len,
                MODEL,
            )
        )
        if not result.success:
            raise RuntimeError(result.error)
        measurements.append(
            {
                "profile": profile,
                "prompt_sha256": prompt_sha256,
                "prompt_tokens": result.reported_prompt_len,
                "completion_tokens": result.output_len,
                "ttft_ms": result.ttft * 1000,
                "mean_itl_ms": float(np.mean(result.itl)) * 1000,
                "median_itl_ms": float(np.median(result.itl)) * 1000,
                "e2e_latency_ms": result.latency * 1000,
            }
        )
    OUTPUT.write_text(
        json.dumps(
            {
                "format": "SGLANG-QWEN35-COLD-WARM-CHARACTERIZATION-v1",
                "gpu_uuid": "GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96",
                "seed": 0,
                "request_count": 2,
                "trace_dataset_member": False,
                "measurements": measurements,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
