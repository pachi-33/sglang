"""Measure M5 merged Qwen3.5 projections on the required V100."""

from __future__ import annotations

import fcntl
import json
import os
import statistics
from pathlib import Path

import torch

from sglang.srt.layers.qwen3_5.checkpoint import Qwen35Checkpoint
from sglang.srt.layers.qwen3_5.dense import linear_fp16
from sglang.srt.layers.qwen3_5.quantization import linear_fp8, quantize_fp8

ROOT = Path(
    os.environ.get(
        "QWEN35_MODEL_DIR",
        "/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16",
    )
)
OUT = Path(
    os.environ.get(
        "QWEN35_PROJECTION_BENCH_OUTPUT",
        "test/Qwen3_5MoECompat/reports/projection_packing_benchmark.json",
    )
)
SIZES = (1, 4, 32, 128, 512, 2048)


def _time(fn, repeats: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _kernel_count(fn) -> tuple[int, dict[str, int]]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        fn()
        torch.cuda.synchronize()
    histogram: dict[str, int] = {}
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            histogram[event.name] = histogram.get(event.name, 0) + 1
    return sum(histogram.values()), histogram


def _individual_gdn(x, w):
    a8 = quantize_fp8(x)
    return (
        linear_fp8(a8, w["linear_attn.in_proj_qkv"]),
        linear_fp8(a8, w["linear_attn.in_proj_z"]),
        linear_fp16(x, w["linear_attn.in_proj_b"]),
        linear_fp16(x, w["linear_attn.in_proj_a"]),
    )


def _merged_gdn(x, w):
    a8 = quantize_fp8(x)
    qkv_z = linear_fp8(a8, w["linear_attn.in_proj_qkv_z"])
    ba = linear_fp16(x, w["linear_attn.in_proj_ba"])
    return qkv_z[:, :8192], qkv_z[:, 8192:], ba[:, :32], ba[:, 32:]


def _individual_full(x, w):
    a8 = quantize_fp8(x)
    return tuple(
        linear_fp8(a8, w[name])
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    )


def _merged_full(x, w):
    a8 = quantize_fp8(x)
    qkv = linear_fp8(a8, w["self_attn.qkv_proj"])
    return qkv[:, :8192], qkv[:, 8192:8704], qkv[:, 8704:]


def _assert_exact(label: str, old, new) -> None:
    for index, (before, after) in enumerate(zip(old, new)):
        if not torch.equal(before, after):
            difference = (before.float() - after.float()).abs().max().item()
            raise AssertionError(f"{label} component {index} differs; max={difference}")


def main() -> None:
    required_uuid = "GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96"
    if required_uuid not in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to the required V100 UUID")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("projection benchmark requires the SM70 V100")
    lock = open("/tmp/qwen35-v100-gpu.lock", "a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    loader = Qwen35Checkpoint(ROOT)
    records = []
    for layer_id, kind, individual, merged in (
        (0, "gdn", _individual_gdn, _merged_gdn),
        (3, "full", _individual_full, _merged_full),
    ):
        weights = loader.load_layer(layer_id, device="cuda")
        for tokens in SIZES:
            torch.manual_seed(20260907 + layer_id * 10_000 + tokens)
            x = (
                torch.randn((tokens, 2048), device="cuda", dtype=torch.float16) * 0.1
            ).contiguous()
            _assert_exact(
                f"layer {layer_id}, T={tokens}",
                individual(x, weights),
                merged(x, weights),
            )
            old_ms, new_ms = _time(lambda: individual(x, weights)), _time(
                lambda: merged(x, weights)
            )
            old_kernels, old_histogram = _kernel_count(lambda: individual(x, weights))
            new_kernels, new_histogram = _kernel_count(lambda: merged(x, weights))
            records.append(
                {
                    "layer": layer_id,
                    "kind": kind,
                    "tokens": tokens,
                    "individual_ms": old_ms,
                    "merged_ms": new_ms,
                    "individual_cuda_kernels": old_kernels,
                    "merged_cuda_kernels": new_kernels,
                    "individual_kernel_histogram": old_histogram,
                    "merged_kernel_histogram": new_histogram,
                    "bitwise_fp16_outputs": True,
                }
            )
        del weights
        torch.cuda.empty_cache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "environment": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "capability": torch.cuda.get_device_capability(),
                    "gpu": torch.cuda.get_device_name(),
                    "uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                "repeats": 10,
                "semantics": "same FP16 input; includes shared A8 quantization and GDN B/A; excludes loading and compilation",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
