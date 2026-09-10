#!/usr/bin/env python3
"""Benchmark O_DIRECT SSD -> pinned CPU -> one CUDA GPU bandwidth."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import queue
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT = "SGLANG-SSD-CPU-GPU-BANDWIDTH-v1"
ALIGNMENT = 4096
libc = ctypes.CDLL(None, use_errno=True)
libc.pread.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_longlong,
]
libc.pread.restype = ctypes.c_ssize_t


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    try:
        with partial.open("xb") as file:
            file.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
            file.flush()
            os.fsync(file.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    try:
        with partial.open("xb") as file:
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _aligned_pinned(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    storage = torch.empty(length + ALIGNMENT, dtype=torch.uint8, pin_memory=True)
    shift = (-storage.data_ptr()) % ALIGNMENT
    view = storage[shift : shift + length]
    if view.numel() != length or view.data_ptr() % ALIGNMENT:
        raise RuntimeError("failed to create an O_DIRECT-aligned pinned buffer")
    return storage, view


def _pread_exact(fd: int, address: int, length: int, offset: int) -> None:
    done = 0
    while done < length:
        read_bytes = libc.pread(fd, address + done, length - done, offset + done)
        if read_bytes < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if read_bytes == 0:
            raise OSError(f"short direct read at offset {offset + done}")
        done += int(read_bytes)


def _direct_fd(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECT)


def _summary(seconds: list[float], bytes_per_sample: int) -> dict[str, Any]:
    values = np.asarray(seconds, dtype=np.float64)
    percentiles = np.percentile(values, [50, 90, 95, 99])
    p50 = float(percentiles[0])
    return {
        "samples": len(seconds),
        "bytes_per_sample": bytes_per_sample,
        "mean_seconds": float(values.mean()),
        "p50_seconds": p50,
        "p90_seconds": float(percentiles[1]),
        "p95_seconds": float(percentiles[2]),
        "p99_seconds": float(percentiles[3]),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
        "GB_per_second_at_p50": bytes_per_sample / p50 / 1e9,
        "GiB_per_second_at_p50": bytes_per_sample / p50 / (1 << 30),
    }


def _ssd_to_cpu(
    path: Path, total_bytes: int, chunk_bytes: int, rounds: int
) -> tuple[list[float], bool]:
    storage, host = _aligned_pinned(chunk_bytes)
    fd = _direct_fd(path)
    try:
        # Untimed warm-up: 256 MiB or the whole file, whichever is smaller.
        warmup_bytes = min(total_bytes, 256 << 20)
        for offset in range(0, warmup_bytes, chunk_bytes):
            _pread_exact(fd, host.data_ptr(), chunk_bytes, offset)
        observed: list[float] = []
        for _ in range(rounds):
            start = time.perf_counter_ns()
            for offset in range(0, total_bytes, chunk_bytes):
                _pread_exact(fd, host.data_ptr(), chunk_bytes, offset)
            observed.append((time.perf_counter_ns() - start) / 1e9)
        content_is_zero = int(torch.count_nonzero(host).item()) == 0
        return observed, content_is_zero
    finally:
        os.close(fd)
        del host, storage


def _active_link_snapshot() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pstate,pcie.link.gen.current,"
            "pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _cpu_to_gpu(
    total_bytes: int, buffer_bytes: int, rounds: int
) -> tuple[list[float], str, bool, int]:
    if total_bytes % buffer_bytes:
        raise ValueError("H2D bytes per round must be divisible by the buffer size")
    repetitions = total_bytes // buffer_bytes
    storage, host = _aligned_pinned(buffer_bytes)
    host.zero_()
    device = torch.empty(buffer_bytes, dtype=torch.uint8, device="cuda:0")
    stream = torch.cuda.Stream(device="cuda:0")
    for _ in range(3):
        with torch.cuda.stream(stream):
            device.copy_(host, non_blocking=True)
    stream.synchronize()

    observed: list[float] = []
    active_link = ""
    for round_index in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(repetitions):
                device.copy_(host, non_blocking=True)
            end.record(stream)
        if round_index == 0:
            active_link = _active_link_snapshot()
        end.synchronize()
        observed.append(float(start.elapsed_time(end)) / 1000.0)
    peak_allocated = int(torch.cuda.max_memory_allocated("cuda:0"))
    content_is_zero = int(torch.count_nonzero(device).item()) == 0
    del device, host, storage
    torch.cuda.empty_cache()
    return observed, active_link, content_is_zero, peak_allocated


def _pipeline_once(
    path: Path,
    total_bytes: int,
    chunk_bytes: int,
    pinned: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.Tensor,
    stream: torch.cuda.Stream,
) -> tuple[float, float]:
    free_buffers: queue.Queue[int] = queue.Queue()
    filled_buffers: queue.Queue[tuple[int, int] | None] = queue.Queue()
    for index in range(len(pinned)):
        free_buffers.put(index)
    errors: list[BaseException] = []
    read_busy_ns = 0

    def reader() -> None:
        nonlocal read_busy_ns
        fd = _direct_fd(path)
        try:
            for offset in range(0, total_bytes, chunk_bytes):
                index = free_buffers.get()
                started = time.perf_counter_ns()
                _pread_exact(fd, pinned[index][1].data_ptr(), chunk_bytes, offset)
                read_busy_ns += time.perf_counter_ns() - started
                filled_buffers.put((index, chunk_bytes))
        except BaseException as error:
            errors.append(error)
        finally:
            os.close(fd)
            filled_buffers.put(None)

    stream.synchronize()
    worker = threading.Thread(target=reader, name="direct-ssd-reader")
    start = time.perf_counter_ns()
    worker.start()
    with torch.cuda.stream(stream):
        while True:
            item = filled_buffers.get()
            if item is None:
                break
            index, length = item
            device[:length].copy_(pinned[index][1][:length], non_blocking=True)
            stream.synchronize()
            free_buffers.put(index)
    worker.join()
    elapsed = (time.perf_counter_ns() - start) / 1e9
    if errors:
        raise errors[0]
    return elapsed, read_busy_ns / 1e9


def _ssd_to_gpu(
    path: Path,
    total_bytes: int,
    chunk_bytes: int,
    buffer_count: int,
    rounds: int,
) -> tuple[list[float], list[float], bool, int]:
    # Exclude the preceding H2D correctness reduction's temporary workspace.
    torch.cuda.reset_peak_memory_stats(0)
    pinned = [_aligned_pinned(chunk_bytes) for _ in range(buffer_count)]
    device = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda:0")
    stream = torch.cuda.Stream(device="cuda:0")
    _pipeline_once(
        path,
        min(total_bytes, 256 << 20),
        chunk_bytes,
        pinned,
        device,
        stream,
    )
    observed = []
    read_busy = []
    for _ in range(rounds):
        elapsed, busy = _pipeline_once(
            path, total_bytes, chunk_bytes, pinned, device, stream
        )
        observed.append(elapsed)
        read_busy.append(busy)
    peak_allocated = int(torch.cuda.max_memory_allocated("cuda:0"))
    content_is_zero = int(torch.count_nonzero(device).item()) == 0
    del device, pinned
    torch.cuda.empty_cache()
    return observed, read_busy, content_is_zero, peak_allocated


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.expected_gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly the expected GPU UUID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU is required")
    if torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("the visible GPU is not an SM70 V100")
    if args.rounds < 3:
        raise ValueError("at least three measured rounds are required")

    total_bytes = args.bytes_per_round_gib << 30
    chunk_bytes = args.ssd_chunk_mib << 20
    h2d_buffer_bytes = args.h2d_buffer_mib << 20
    file_size = args.file.stat().st_size
    if file_size != total_bytes:
        raise ValueError(f"test file is {file_size} bytes; expected {total_bytes}")
    if total_bytes % chunk_bytes:
        raise ValueError("file size must be divisible by the SSD chunk size")

    torch.cuda.reset_peak_memory_stats(0)
    ssd_seconds, ssd_correct = _ssd_to_cpu(
        args.file, total_bytes, chunk_bytes, args.rounds
    )
    h2d_seconds, active_link, h2d_correct, h2d_peak = _cpu_to_gpu(
        total_bytes, h2d_buffer_bytes, args.rounds
    )
    pipeline_seconds, pipeline_read_busy, pipeline_correct, pipeline_peak = (
        _ssd_to_gpu(
            args.file,
            total_bytes,
            chunk_bytes,
            args.pipeline_buffers,
            args.rounds,
        )
    )

    ssd_summary = _summary(ssd_seconds, total_bytes)
    h2d_summary = _summary(h2d_seconds, total_bytes)
    pipeline_summary = _summary(pipeline_seconds, total_bytes)
    ratio = (
        pipeline_summary["GB_per_second_at_p50"]
        / ssd_summary["GB_per_second_at_p50"]
    )
    correctness = {
        "ssd_last_buffer_zero": ssd_correct,
        "h2d_destination_zero": h2d_correct,
        "pipeline_destination_zero": pipeline_correct,
        "all_passed": ssd_correct and h2d_correct and pipeline_correct,
    }
    acceptance = {
        "active_link_is_gen3_x16": ", 3, 16," in f", {active_link},",
        "h2d_above_10_GBps": h2d_summary["GB_per_second_at_p50"] > 10.0,
        "pipeline_at_least_90_percent_of_ssd": ratio >= 0.90,
    }
    acceptance["all_passed"] = all(acceptance.values()) and correctness["all_passed"]

    _atomic_npz(
        args.output_raw,
        format=np.asarray(FORMAT),
        total_bytes=np.asarray(total_bytes, dtype=np.int64),
        ssd_seconds=np.asarray(ssd_seconds, dtype=np.float64),
        h2d_seconds=np.asarray(h2d_seconds, dtype=np.float64),
        pipeline_seconds=np.asarray(pipeline_seconds, dtype=np.float64),
        pipeline_ssd_read_busy_seconds=np.asarray(
            pipeline_read_busy, dtype=np.float64
        ),
    )
    raw_samples = {
        "path": str(args.output_raw.resolve()),
        "size": args.output_raw.resolve().stat().st_size,
        "sha256": _sha256(args.output_raw.resolve()),
    }
    result: dict[str, Any] = {
        "format": FORMAT,
        "scope": "synthetic maximum-bandwidth microbenchmark; no model or prefetch",
        "gpu": {
            "expected_uuid": args.expected_gpu_uuid,
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "active_link_snapshot": active_link,
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "configuration": {
            "test_file": str(args.file.resolve()),
            "file_bytes": file_size,
            "bytes_per_round": total_bytes,
            "rounds": args.rounds,
            "ssd_io": "O_DIRECT synchronous pread",
            "ssd_chunk_bytes": chunk_bytes,
            "h2d_buffer_bytes": h2d_buffer_bytes,
            "pipeline_pinned_buffers": args.pipeline_buffers,
            "pipeline_pinned_bytes": args.pipeline_buffers * chunk_bytes,
        },
        "paths": {
            "ssd_to_pinned_cpu": ssd_summary,
            "pinned_cpu_to_gpu": h2d_summary,
            "ssd_to_pinned_cpu_to_gpu_pipeline": pipeline_summary,
        },
        "pipeline_ssd_read_busy_seconds": pipeline_read_busy,
        "pipeline_to_isolated_ssd_bandwidth_ratio": ratio,
        "peak_gpu_allocated_bytes": max(h2d_peak, pipeline_peak),
        "correctness": correctness,
        "acceptance": acceptance,
        "raw_samples": raw_samples,
    }
    _atomic_json(args.output_summary, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output-raw", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--ssd-chunk-mib", type=int, default=16)
    parser.add_argument("--h2d-buffer-mib", type=int, default=256)
    parser.add_argument("--pipeline-buffers", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--bytes-per-round-gib", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    try:
        result = benchmark(_parse_args())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
