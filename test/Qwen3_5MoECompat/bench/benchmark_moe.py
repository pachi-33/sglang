"""Repeatable V100 routed-MoE benchmark (run as a module)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from test.Qwen3_5MoECompat.unit.test_environment import V100_GPU_LOCK_PATH

import torch

# Use a process-private cache so the PTX evidence below belongs to this run.
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/qwen35-moe-bench-triton-{os.getpid()}")

from sglang.srt.layers.qwen3_5.checkpoint import Qwen35Checkpoint
from sglang.srt.layers.qwen3_5.moe import (
    _fp16_add,
    _fp16_linear,
    _fp16_sigmoid_multiply,
    _fp16_swiglu,
    execute_experts,
    execute_experts_unfused_baseline,
    route_topk,
)

ROOT = Path("/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16")
# CI/tuning runs can retain independent reports without modifying the
# canonical cap-320 artifact.
OUT = Path(
    os.environ.get(
        "QWEN35_MOE_BENCH_OUTPUT", "test/Qwen3_5MoECompat/reports/moe_benchmark.json"
    )
)
SIZES = (1, 4, 32, 128, 512, 2048)


def _time(fn, repeats: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def _kernel_profile(fn) -> dict[str, int]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        fn()
        torch.cuda.synchronize()
    histogram: dict[str, int] = {}
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            histogram[event.name] = histogram.get(event.name, 0) + 1
    return histogram


def _ptx_evidence() -> dict[str, bool]:
    ptx = list(Path(os.environ["TRITON_CACHE_DIR"]).rglob("*.ptx"))
    text = "\n".join(path.read_text(errors="ignore") for path in ptx)
    return {
        "current_process_ptx": bool(ptx),
        "target_sm70": ".target sm_70" in text,
        "has_hmma": "mma.sync" in text,
    }


def _full_path(
    x, layer, ids=None, probs=None, *, baseline: bool, residual: torch.Tensor
):
    logits = _fp16_linear(x, layer["mlp.gate"])
    routed_ids, routed_probs = route_topk(logits)
    ids, probs = (routed_ids, routed_probs) if ids is None else (ids, probs)
    shared = _fp16_swiglu(
        x, layer["mlp.shared_expert.gate_up_proj"], layer["mlp.shared_expert.down_proj"]
    )
    shared = _fp16_sigmoid_multiply(
        shared, _fp16_linear(x, layer["mlp.shared_expert_gate"])
    )
    if not baseline:
        return execute_experts(
            x,
            layer["mlp.experts.gate_up_proj"],
            layer["mlp.experts.down_proj"],
            ids,
            probs,
            shared=shared,
            residual=residual,
        )
    route = execute_experts_unfused_baseline(
        x, layer["mlp.experts.gate_up_proj"], layer["mlp.experts.down_proj"], ids, probs
    )
    return _fp16_add(_fp16_add(route, shared), residual)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    required_uuid = "GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96"
    if required_uuid not in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to the required V100 UUID")
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError(f"expected SM70, got {torch.cuda.get_device_capability()}")
    lock = open(V100_GPU_LOCK_PATH, "a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    layer = Qwen35Checkpoint(ROOT).load_layer(1, device="cuda")
    records = []
    torch.manual_seed(20260907)
    for pattern in ("natural_router", "balanced_forced", "eight_hot_forced"):
        for tokens in SIZES:
            x = torch.randn((tokens, 2048), dtype=torch.float16, device="cuda") * 0.1
            residual = torch.randn_like(x) * 0.01
            ids = probs = None
            if pattern == "balanced_forced":
                ids = (
                    torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(
                        tokens, 8
                    )
                    % 256
                )
                probs = torch.full(
                    (tokens, 8), 0.125, dtype=torch.float32, device="cuda"
                )
            elif pattern == "eight_hot_forced":
                ids = (
                    torch.arange(8, device="cuda", dtype=torch.int32)[None, :].expand(
                        tokens, -1
                    )
                ).contiguous()
                probs = torch.full(
                    (tokens, 8), 0.125, dtype=torch.float32, device="cuda"
                )
            fused_fn = lambda: _full_path(
                x, layer, ids, probs, baseline=False, residual=residual
            )
            baseline_fn = lambda: _full_path(
                x, layer, ids, probs, baseline=True, residual=residual
            )
            fused = fused_fn()
            baseline = baseline_fn()
            torch.cuda.synchronize()
            nrmse = (
                (fused.float() - baseline.float()).square().mean().sqrt()
                / baseline.float().square().mean().sqrt()
            ).item()
            if not torch.isfinite(fused).all() or nrmse > 5e-3:
                raise AssertionError((pattern, tokens, nrmse))
            torch.cuda.reset_peak_memory_stats()
            fused_ms = _time(fused_fn)
            fused_peak = torch.cuda.max_memory_allocated()
            fused_kernels = _kernel_profile(fused_fn)
            torch.cuda.reset_peak_memory_stats()
            baseline_ms = _time(baseline_fn)
            baseline_peak = torch.cuda.max_memory_allocated()
            baseline_kernels = _kernel_profile(baseline_fn)
            records.append(
                dict(
                    tokens=tokens,
                    pattern=pattern,
                    fused_ms=fused_ms,
                    unfused_ms=baseline_ms,
                    fused_peak_bytes=fused_peak,
                    unfused_peak_bytes=baseline_peak,
                    fused_cuda_kernels=sum(fused_kernels.values()),
                    unfused_cuda_kernels=sum(baseline_kernels.values()),
                    fused_kernel_histogram=fused_kernels,
                    unfused_kernel_histogram=baseline_kernels,
                    nrmse=nrmse,
                    finite=True,
                )
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            dict(
                environment=dict(
                    torch=torch.__version__,
                    triton=__import__("triton").__version__,
                    cuda=torch.version.cuda,
                    capability=torch.cuda.get_device_capability(),
                    gpu=torch.cuda.get_device_name(),
                    uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    commit=os.popen("git rev-parse HEAD").read().strip(),
                    source_hash=hashlib.sha1(
                        Path("python/sglang/srt/layers/qwen3_5/moe.py").read_bytes()
                        + Path(
                            "python/sglang/srt/layers/qwen3_5/kernels/moe.py"
                        ).read_bytes()
                    ).hexdigest(),
                ),
                backend=dict(
                    runtime="Triton CUDA",
                    profiler="torch.profiler CUDA events",
                    ptx=_ptx_evidence(),
                ),
                layer=1,
                cta_cap=int(os.environ.get("SGLANG_QWEN35_MOE_CTA_CAP", "320")),
                shapes=dict(hidden=2048, intermediate=512, experts=256, top_k=8),
                repeats=10,
                semantics="includes quant/routing/GEMM1/SwiGLU/A4/GEMM2/combine; excludes load and compilation",
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
