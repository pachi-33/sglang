"""Manual single-V100 acceptance for the Qwen3.5 ExpertPack backend.

The command runs the public greedy backend, an A/B/A reset check, and an
independent 2048-token prefill in one process.  It writes machine-readable
evidence without involving the legacy two-GPU pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sglang.srt.layers.qwen3_5.pipeline import load_tokenizer_compat
from sglang.srt.layers.qwen3_5.single_gpu import (
    EXPERT_PACK_MANIFEST_DEFAULT,
    Qwen35SingleGPU,
    _cuda_memory_stats,
)

MODEL_DIR = Path("/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16")
HELLO_ORACLE = [11, 271, 40, 1044, 4313, 310, 958, 279]
ONE_GIB = 1 << 30


def run_acceptance(
    *,
    model_dir: Path,
    manifest_path: Path,
    cache_mib: int,
    output_path: Path | None,
) -> dict[str, object]:
    tokenizer = load_tokenizer_compat(model_dir)
    prompt_a = tokenizer.encode("Hello", add_special_tokens=False)
    prompt_b = tokenizer.encode(
        "Write one sentence about Volta.", add_special_tokens=False
    )
    started = time.time()
    with Qwen35SingleGPU(
        model_dir,
        expert_pack_manifest=manifest_path,
        expert_cache_mib=cache_mib,
        capacity=2048,
    ) as backend:
        generated_a1 = backend.generate_ids(prompt_a, max_new_tokens=8)
        cold = backend.stats["last_generation"]
        generated_b = backend.generate_ids(prompt_b, max_new_tokens=8)
        warm_b = backend.stats["last_generation"]
        generated_a2 = backend.generate_ids(prompt_a, max_new_tokens=8)
        warm_a = backend.stats["last_generation"]

        torch.cuda.reset_peak_memory_stats(backend.device)
        long_ids = (
            torch.arange(2048, device=backend.device, dtype=torch.int32) % 10_000
        ) + 1
        long_started = time.perf_counter_ns()
        with torch.inference_mode():
            long_hidden = backend.runner.embed(long_ids)
            long_hidden = backend.runner.prefill_hidden(
                long_hidden, cache=backend.cache
            )
        long_finished = time.perf_counter_ns()
        long_finite = bool(torch.isfinite(long_hidden).all().item())
        long_memory = _cuda_memory_stats(backend.device)

        capacity_error = None
        try:
            decode_id = torch.tensor([1], device=backend.device, dtype=torch.int32)
            backend.runner.decode_hidden(
                backend.runner.embed(decode_id),
                cache=backend.cache,
                expected_prefix_len=2048,
            )
        except RuntimeError as error:
            capacity_error = str(error)
        if capacity_error is None or "capacity" not in capacity_error:
            raise AssertionError("decode at full capacity did not fail explicitly")
        if backend.cache.poisoned:
            raise AssertionError("capacity rejection poisoned the request cache")
        backend.runner.reset_request_cache(backend.cache)

        generated_after_capacity = backend.generate_ids(prompt_a, max_new_tokens=8)
        final_stats = backend.stats
        cache_bytes = int(final_stats["cache_capacity_bytes"])
        margin = int(long_memory["peak_reserved_margin_bytes"])
        report: dict[str, object] = {
            "status": "passed",
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "device": {
                "visible_count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(backend.device),
                "capability": list(torch.cuda.get_device_capability(backend.device)),
            },
            "profile": {
                "cache_mib": cache_mib,
                "capacity": 2048,
                "stage_slots": 16,
                "io_workers": 2,
            },
            "hello": {
                "oracle": HELLO_ORACLE,
                "first": generated_a1,
                "second": generated_a2,
                "after_capacity": generated_after_capacity,
                "oracle_exact": generated_a1 == HELLO_ORACLE,
                "aba_exact": generated_a1 == generated_a2,
                "post_capacity_exact": generated_a1 == generated_after_capacity,
            },
            "prompt_b_tokens": len(prompt_b),
            "prompt_b_completion": generated_b,
            "timing": {"cold_a": cold, "warm_b": warm_b, "warm_a": warm_a},
            "context_2048": {
                "prefill_ms": (long_finished - long_started) / 1e6,
                "finite": long_finite,
                "capacity_error": capacity_error,
                "memory": long_memory,
            },
            "memory_gate": {
                "cache_bytes": cache_bytes,
                "cache_within_budget": cache_bytes <= cache_mib * (1 << 20),
                "required_margin_bytes": ONE_GIB,
                "actual_margin_bytes": margin,
                "margin_passed": margin >= ONE_GIB,
            },
            "store": {
                key: value
                for key, value in final_stats.items()
                if key not in {"last_generation"}
            },
        }
        checks = (
            generated_a1 == HELLO_ORACLE,
            generated_a1 == generated_a2 == generated_after_capacity,
            long_finite,
            cache_bytes <= cache_mib * (1 << 20),
            margin >= ONE_GIB,
            final_stats.get("state") == "READY",
        )
        if not all(checks):
            report["status"] = "failed"

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output_path)
    print(encoded, end="")
    if report["status"] != "passed":
        raise AssertionError("single-V100 ExpertPack acceptance failed")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--manifest", type=Path, default=Path(EXPERT_PACK_MANIFEST_DEFAULT)
    )
    parser.add_argument("--cache-mib", type=int, choices=(7168, 6144), default=7168)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_acceptance(
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        cache_mib=args.cache_mib,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
