#!/usr/bin/env python3
"""Measure exact ExpertPack payload copies from pinned CPU memory to one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang.srt.layers.qwen3_5.expert_pack.format import PAYLOAD_SIZE, load_manifest
from sglang.srt.layers.qwen3_5.expert_pack.store import _COMPONENT_LAYOUT

H2D_FORMAT = "SGLANG-QWEN35-PINNED-H2D-BENCHMARK-v1"


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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite H2D output: {path}")
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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite H2D summary: {path}")
    try:
        with partial.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _load_real_payloads(
    manifest_path: Path, count: int
) -> tuple[torch.Tensor, str, list[str]]:
    manifest = load_manifest(manifest_path, verify_identity=True)
    pack_path = Path(manifest.pack_path(manifest_path)).resolve()
    staging = torch.empty((count, PAYLOAD_SIZE), dtype=torch.uint8, pin_memory=True)
    digests: list[str] = []
    fd = os.open(pack_path, os.O_RDONLY)
    try:
        for expert_id in range(count):
            record = manifest.record(1, expert_id)
            row = staging[expert_id]
            view = memoryview(row.numpy()).cast("B")
            read_bytes = os.preadv(fd, [view], int(record.offset))
            if read_bytes != PAYLOAD_SIZE:
                raise OSError(
                    f"short ExpertPack read for layer 1 expert {expert_id}: "
                    f"{read_bytes} != {PAYLOAD_SIZE}"
                )
            digest = hashlib.sha256(view).hexdigest()
            if digest != record.sha256:
                raise ValueError(
                    f"ExpertPack checksum mismatch for layer 1 expert {expert_id}"
                )
            digests.append(digest)
    finally:
        os.close(fd)
    return staging, str(pack_path), digests


def _host_component(
    staging: torch.Tensor,
    expert_index: int,
    offset: int,
    nbytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = staging[expert_index].narrow(0, offset, nbytes)
    if dtype == torch.float32:
        value = value.view(torch.float32)
    return value.reshape(shape or ())


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    candidates = tuple(args.candidates)
    if (
        not candidates
        or tuple(sorted(set(candidates))) != candidates
        or candidates[0] <= 0
    ):
        raise ValueError("candidates must be unique ascending positive integers")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.expected_gpu_uuid:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly the expected V100 UUID"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("pinned H2D benchmark requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"expected SM70 V100, got SM{capability[0]}{capability[1]}")

    maximum = candidates[-1]
    staging, pack_path, source_digests = _load_real_payloads(
        args.manifest_path.resolve(), maximum
    )
    destination = {
        name: torch.empty((maximum, *shape), dtype=dtype, device=device)
        for name, _, _, shape, dtype in _COMPONENT_LAYOUT
    }
    stream = torch.cuda.Stream(device=device)
    events = {
        count: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for count in candidates
    }

    def copy_once(count: int) -> float:
        start, end = events[count]
        with torch.cuda.stream(stream):
            start.record(stream)
            for expert_index in range(count):
                for name, offset, nbytes, shape, dtype in _COMPONENT_LAYOUT:
                    destination[name][expert_index].copy_(
                        _host_component(
                            staging,
                            expert_index,
                            offset,
                            nbytes,
                            shape,
                            dtype,
                        ),
                        non_blocking=True,
                    )
            end.record(stream)
        end.synchronize()
        return float(start.elapsed_time(end))

    for count in candidates:
        for _ in range(args.warmup):
            copy_once(count)

    observed: dict[int, list[float]] = {count: [] for count in candidates}
    for repetition in range(args.samples):
        offset = repetition % len(candidates)
        order = candidates[offset:] + candidates[:offset]
        for count in order:
            observed[count].append(copy_once(count))
    torch.cuda.synchronize(device)

    latency_ms = np.asarray([observed[count] for count in candidates], dtype=np.float64)
    if latency_ms.shape != (len(candidates), args.samples):
        raise RuntimeError("H2D sample matrix has an invalid shape")
    if not np.all(np.isfinite(latency_ms)) or np.any(latency_ms <= 0.0):
        raise RuntimeError("H2D benchmark produced invalid durations")
    candidate_array = np.asarray(candidates, dtype=np.int32)
    bytes_array = candidate_array.astype(np.int64) * PAYLOAD_SIZE
    _atomic_npz(
        args.output_npz,
        format=np.asarray(H2D_FORMAT),
        candidate_counts=candidate_array,
        transfer_bytes=bytes_array,
        latency_ms=latency_ms,
        payload_size=np.asarray(PAYLOAD_SIZE, dtype=np.int64),
        warmup=np.asarray(args.warmup, dtype=np.int32),
        samples=np.asarray(args.samples, dtype=np.int32),
        source_payload_sha256=np.asarray(source_digests, dtype="<U64"),
    )

    rows = []
    for index, count in enumerate(candidates):
        samples = latency_ms[index]
        percentiles = np.percentile(samples, [50, 90, 95, 99])
        bytes_value = int(bytes_array[index])
        rows.append(
            {
                "candidate_count": count,
                "transfer_bytes": bytes_value,
                "samples": args.samples,
                "mean_ms": float(samples.mean()),
                "p50_ms": float(percentiles[0]),
                "p90_ms": float(percentiles[1]),
                "p95_ms": float(percentiles[2]),
                "p99_ms": float(percentiles[3]),
                "effective_gib_per_second_at_p50": (
                    bytes_value / float(percentiles[0]) * 1000.0 / (1 << 30)
                ),
            }
        )
    summary: dict[str, Any] = {
        "format": H2D_FORMAT,
        "scope": "pinned CPU to V100 only; excludes SSD, checksum, and allocation",
        "gpu": {
            "uuid": args.expected_gpu_uuid,
            "name": torch.cuda.get_device_name(device),
            "capability": list(capability),
        },
        "manifest_path": str(args.manifest_path.resolve()),
        "pack_path": pack_path,
        "payload_size": PAYLOAD_SIZE,
        "component_copies_per_expert": len(_COMPONENT_LAYOUT),
        "warmup_per_candidate": args.warmup,
        "samples_per_candidate": args.samples,
        "rows": rows,
        "raw_samples": {
            "file": str(args.output_npz.resolve()),
            "size": args.output_npz.resolve().stat().st_size,
            "sha256": _sha256(args.output_npz.resolve()),
        },
    }
    _atomic_json(args.output_summary, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--candidates", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = benchmark(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
