#!/usr/bin/env python3
"""Profile per-layer single-token decode timing in the resident PP runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from sglang.bench_serving import sample_random_requests
from sglang.srt.layers.qwen3_5.pipeline import (
    DEFAULT_BACK_UUID,
    DEFAULT_FRONT_UUID,
    DEFAULT_SPLIT_LAYER,
    Qwen35Pipeline,
    load_tokenizer_compat,
)

PROFILE_FORMAT = "SGLANG-QWEN35-RESIDENT-DECODE-PROFILE-v1"


def _atomic_write(path: Path, data: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {path}")
    try:
        with partial.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(partial, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _prompt_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(token_ids)}i", *token_ids)).hexdigest()


def _reproduce_prompt(
    *,
    model_dir: Path,
    dataset_path: Path,
    seed: int,
    request_index: int,
    input_tokens: int,
    output_tokens: int,
) -> list[int]:
    random.seed(seed)
    np.random.seed(seed)
    tokenizer = load_tokenizer_compat(model_dir)
    requests = sample_random_requests(
        input_len=input_tokens,
        output_len=output_tokens,
        num_prompts=max(128, request_index + 1),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=str(dataset_path),
        return_token_ids=True,
    )
    prompt, reported_input, reported_output = requests[request_index]
    if not isinstance(prompt, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in prompt
    ):
        raise RuntimeError("reproduced prompt is not an integer token ID list")
    if reported_input != input_tokens or reported_output != output_tokens:
        raise RuntimeError("reproduced request dimensions do not match the contract")
    return prompt


def _load_oracle(path: Path, request_index: int, expected_tokens: int) -> list[int]:
    with np.load(path, allow_pickle=False) as data:
        if "sampled_token_ids" not in data:
            raise ValueError("oracle NPZ does not contain sampled_token_ids")
        values = data["sampled_token_ids"]
        if values.ndim != 2 or request_index >= values.shape[0]:
            raise ValueError("oracle sampled_token_ids has an invalid request axis")
        if values.shape[1] < expected_tokens or values.dtype != np.int32:
            raise ValueError("oracle sampled_token_ids has an invalid token axis")
        return [int(value) for value in values[request_index, :expected_tokens]]


def _flatten_timings(
    steps: list[dict[str, Any]], expected_layer_ids: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decode_index, step in enumerate(steps):
        if step.get("step_id") != decode_index + 1:
            raise RuntimeError("decode timing step IDs are not contiguous")
        if step.get("role") != "back":
            raise RuntimeError("decode timing came from the wrong PP role")
        layers = step.get("layers")
        if not isinstance(layers, list) or len(layers) != len(expected_layer_ids):
            raise RuntimeError("decode timing has an incomplete layer slice")
        for expected_layer, layer in zip(expected_layer_ids, layers):
            if layer.get("layer_id") != expected_layer:
                raise RuntimeError("decode timing layer IDs are not contiguous")
            start = float(layer["layer_start_ms"])
            router_start = float(layer["router_start_ms"])
            router_ready = float(layer["router_ready_ms"])
            routed_start = float(layer["routed_expert_start_ms"])
            end = float(layer["layer_end_ms"])
            if not start <= router_start <= router_ready <= routed_start <= end:
                raise RuntimeError("decode timing boundaries are not monotonic")
            rows.append(
                {
                    "format": PROFILE_FORMAT,
                    "decode_index": decode_index,
                    "step_id": int(step["step_id"]),
                    "prefix_len": int(step["prefix_len"]),
                    "role": "back",
                    "layer_id": expected_layer,
                    "layer_type": (
                        "full_attention" if (expected_layer + 1) % 4 == 0 else "gdn"
                    ),
                    "routed_format": "fp16" if expected_layer == 39 else "nvfp4",
                    "layer_start_ms": start,
                    "router_start_ms": router_start,
                    "router_ready_ms": router_ready,
                    "routed_expert_start_ms": routed_start,
                    "layer_end_ms": end,
                    "pre_router_ms": router_start - start,
                    "router_ms": router_ready - router_start,
                    "shared_gap_ms": routed_start - router_ready,
                    "routed_tail_ms": end - routed_start,
                    "layer_total_ms": end - start,
                }
            )
    return rows


def profile(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir.resolve()
    dataset_path = args.dataset_path.resolve()
    oracle_npz = args.oracle_npz.resolve()
    prompt_ids = _reproduce_prompt(
        model_dir=model_dir,
        dataset_path=dataset_path,
        seed=args.seed,
        request_index=args.request_index,
        input_tokens=args.input_tokens,
        output_tokens=args.oracle_tokens,
    )
    prompt_sha256 = _prompt_sha256(prompt_ids)
    if prompt_sha256 != args.expected_prompt_sha256:
        raise RuntimeError(
            f"prompt SHA-256 mismatch: {prompt_sha256} != "
            f"{args.expected_prompt_sha256}"
        )
    oracle = _load_oracle(
        oracle_npz, args.request_index, expected_tokens=args.oracle_tokens
    )
    output_tokens = args.decode_steps + 1
    if len(prompt_ids) + output_tokens > args.capacity:
        raise ValueError("prompt plus profiled generation exceeds pipeline capacity")

    with Qwen35Pipeline(
        model_dir,
        capacity=args.capacity,
        front_uuid=args.front_uuid,
        back_uuid=args.back_uuid,
        split_layer=args.split_layer,
    ) as pipeline:
        front_info, back_info = pipeline.worker_info
        for info, role in ((front_info, "front"), (back_info, "back")):
            if info.get("expert_offload_enabled") is not False:
                raise RuntimeError(f"{role} worker unexpectedly enabled expert offload")
            if info.get("routed_experts_resident") is not True:
                raise RuntimeError(f"{role} worker did not confirm resident experts")

        warmup = pipeline.generate_ids(
            prompt_ids,
            max_new_tokens=args.warmup_decode_steps + 1,
            eos_token_ids=(),
        )
        control_started = time.perf_counter()
        control = pipeline.generate_ids(
            prompt_ids, max_new_tokens=output_tokens, eos_token_ids=()
        )
        control_seconds = time.perf_counter() - control_started
        control_stats = {
            "front": dict(pipeline.worker_stats[0]),
            "back": dict(pipeline.worker_stats[1]),
        }

        timed_started = time.perf_counter()
        timed = pipeline.generate_ids(
            prompt_ids,
            max_new_tokens=output_tokens,
            eos_token_ids=(),
            decode_timing_role="back",
        )
        timed_seconds = time.perf_counter() - timed_started
        timing_steps = list(pipeline.last_decode_timing)
        timed_stats = {
            "front": dict(pipeline.worker_stats[0]),
            "back": dict(pipeline.worker_stats[1]),
        }

    if len(warmup) != args.warmup_decode_steps + 1:
        raise RuntimeError("warmup did not complete the requested decode steps")
    if control != timed:
        raise RuntimeError("timing changed the generated token IDs")
    if timed[: args.oracle_tokens] != oracle:
        raise RuntimeError("resident PP output does not match the EXP-0001 oracle")
    if len(timing_steps) != args.decode_steps:
        raise RuntimeError(
            f"expected {args.decode_steps} decode timing steps, got "
            f"{len(timing_steps)}"
        )
    expected_layers = tuple(range(args.split_layer, 40))
    raw_rows = _flatten_timings(timing_steps, expected_layers)
    if len(raw_rows) != args.decode_steps * len(expected_layers):
        raise RuntimeError("profile row count does not match steps times layers")
    expected_prefixes = list(
        range(args.input_tokens, args.input_tokens + args.decode_steps)
    )
    actual_prefixes = [int(step["prefix_len"]) for step in timing_steps]
    if actual_prefixes != expected_prefixes:
        raise RuntimeError("decode prefix lengths are not contiguous")

    raw_data = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in raw_rows
    )
    _atomic_write(args.output_jsonl, raw_data)
    summary: dict[str, Any] = {
        "format": PROFILE_FORMAT,
        "model_dir": str(model_dir),
        "dataset_path": str(dataset_path),
        "oracle_npz": str(oracle_npz),
        "request_index": args.request_index,
        "seed": args.seed,
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": prompt_sha256,
        "oracle_tokens_checked": args.oracle_tokens,
        "warmup_decode_steps": args.warmup_decode_steps,
        "measurement_decode_steps": args.decode_steps,
        "generated_tokens": len(timed),
        "token_checks": {
            "timed_equals_control": True,
            "first_oracle_tokens_match_exp0001": True,
        },
        "front_uuid": args.front_uuid,
        "back_uuid": args.back_uuid,
        "split_layer": args.split_layer,
        "front_layer_ids": list(range(args.split_layer)),
        "back_layer_ids": list(expected_layers),
        "worker_info": {"front": front_info, "back": back_info},
        "control_e2e_seconds": control_seconds,
        "timed_e2e_seconds": timed_seconds,
        "timing_wall_overhead_ratio": timed_seconds / control_seconds - 1.0,
        "control_worker_stats": control_stats,
        "timed_worker_stats": timed_stats,
        "timing_rows": len(raw_rows),
        "generated_token_ids": timed,
        "raw_timing_file": str(args.output_jsonl.resolve()),
    }
    _atomic_write(
        args.output_summary,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--oracle-npz", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--front-uuid", default=DEFAULT_FRONT_UUID)
    parser.add_argument("--back-uuid", default=DEFAULT_BACK_UUID)
    parser.add_argument("--split-layer", type=int, default=DEFAULT_SPLIT_LAYER)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--oracle-tokens", type=int, default=256)
    parser.add_argument("--warmup-decode-steps", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument(
        "--expected-prompt-sha256",
        default="4e7bf8d35d419417f9274499ef041fe6cc5de9210c9e1ffc6f05bc8f9af823e8",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = profile(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
