"""Profile a real four-layer stateless Qwen3.5 slice on the required V100."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import statistics
from pathlib import Path

import torch

MODEL = Path("/home/yaozhenyang/huggingface/Qwen-AgentWorld-35B-A3B-NVFP4_fp16")
SIZES = (1, 4, 32, 128, 512, 2048)
UUID = "GPU-49f8dc6e-3362-d9b2-d1da-8755345e8f96"


def _ptx(root: Path) -> tuple[dict[str, bool], set[str]]:
    texts = (
        [p.read_text(errors="ignore") for p in root.rglob("*.ptx")]
        if root.is_dir()
        else []
    )
    text = "\n".join(texts)
    entries = set(re.findall(r"\.entry\s+([A-Za-z0-9_]+)", text))
    return (
        {
            "current_process_ptx": bool(texts),
            "target_sm70": ".target sm_70" in text,
            "has_hmma": "mma.sync" in text,
        },
        entries,
    )


def classify_cuda_event(name: str, triton_entries: set[str]) -> str:
    """Return memory/compute; reject framework or unknown compute launches."""
    lower = name.lower()
    if any(
        word in lower for word in ("memcpy", "memset", "[cuda memcpy]", "[cuda memset]")
    ):
        return "memory"
    if any(
        word in lower for word in ("at::native", "at::", "torch::", "cublas", "cutlass")
    ):
        raise AssertionError(f"framework CUDA compute event: {name}")
    if not any(
        name == entry or name.startswith(entry + "(") for entry in triton_entries
    ):
        raise AssertionError(f"unknown CUDA compute event: {name}")
    return "compute"


def _profile(fn, entries):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        output = fn()
        torch.cuda.synchronize()
    histogram, memory_events, unknown = {}, {}, []
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            try:
                kind = classify_cuda_event(event.name, entries)
            except AssertionError:
                unknown.append(event.name)
                continue
            target = memory_events if kind == "memory" else histogram
            target[event.name] = target.get(event.name, 0) + 1
    if unknown:
        raise AssertionError(f"disallowed CUDA events: {unknown}")
    tensors = output if isinstance(output, tuple) else (output,)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise AssertionError("nonfinite output")
    return histogram, memory_events


def _median_ms(fn, repeats=5):
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="test/Qwen3_5MoECompat/reports/backend_audit.json"
    )
    args = parser.parse_args()
    if (
        UUID not in os.environ.get("CUDA_VISIBLE_DEVICES", "")
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        raise RuntimeError("requires the configured SM70 V100 UUID")
    lock = open("/tmp/qwen35-v100-gpu.lock", "a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    cache_dir = Path(f"/tmp/qwen35-backend-audit-{os.getpid()}")
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    from sglang.srt.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

    model = Qwen3_5MoeForConditionalGeneration.from_checkpoint(
        MODEL, selected_layer_ids=range(4), device="cuda"
    )
    records = []
    for tokens in SIZES:
        torch.manual_seed(20260907 + tokens)
        input_ids = torch.randint(
            0, 248320, (tokens,), device="cuda", dtype=torch.int64
        )
        positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        fn = lambda: model.forward_no_cache(
            input_ids=input_ids, positions=positions, cu_seqlens=cu, max_seqlen=tokens
        )
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latency = _median_ms(fn)
        evidence, entries = _ptx(cache_dir)
        if not entries:
            raise AssertionError(
                "warmup produced no private-process Triton PTX entries"
            )
        histogram, memory = _profile(fn, entries)
        final_hidden, logits = fn()
        if not (torch.isfinite(final_hidden).all() and torch.isfinite(logits).all()):
            raise AssertionError("nonfinite full-model output")
        records.append(
            {
                "tokens": tokens,
                "latency_ms_median": latency,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_kernel_histogram": histogram,
                "cuda_memory_events": memory,
                "final_hidden_shape": list(final_hidden.shape),
                "logits_shape": list(logits.shape),
                "finite": True,
            }
        )
    source_paths = sorted(Path("python/sglang/srt/layers/qwen3_5").rglob("*.py")) + [
        Path("python/sglang/srt/models/qwen3_5_moe.py"),
        Path(__file__),
    ]
    digest = hashlib.sha1()
    for source in source_paths:
        digest.update(str(source).encode())
        digest.update(source.read_bytes())
    evidence, _ = _ptx(cache_dir)
    result = {
        "environment": {
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "capability": torch.cuda.get_device_capability(),
            "gpu": torch.cuda.get_device_name(),
            "uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_hash": digest.hexdigest(),
        },
        "layers": [0, 1, 2, 3],
        "full_no_cache": True,
        "warmup": 3,
        "repeats": 5,
        "private_triton_cache": str(cache_dir),
        "ptx": evidence,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
