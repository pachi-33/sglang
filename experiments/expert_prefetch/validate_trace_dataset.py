#!/usr/bin/env python3
"""Validate and index a Qwen3.5 expert activation trace dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

DATASET_FORMAT = "SGLANG-QWEN35-EXPERT-TRACE-DATASET-v1"
TRACE_FORMAT = "SGLANG-QWEN35-EXPERT-TRACE-v1"
EXPECTED_ARRAYS = {
    "expert_ids": np.dtype("uint8"),
    "sampled_token_ids": np.dtype("int32"),
    "model_input_token_ids": np.dtype("int32"),
    "model_input_positions": np.dtype("int32"),
    "phase": np.dtype("uint8"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bench_record(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1:
        raise ValueError(f"expected one benchmark record, found {len(records)}")
    record = records[0]
    if record.get("expert_trace") is not True:
        raise ValueError("benchmark record is not an expert-trace run")
    return record


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite dataset output: {path}")
    with partial.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_dataset(
    *,
    bench_jsonl: Path,
    trace_dir: Path,
    output_manifest: Path,
    output_summary: Path,
    expected_count: int,
    prompt_tokens: int,
    completion_tokens: int,
    gpu_uuid: str,
    source_dataset: Path | None = None,
) -> dict[str, Any]:
    if expected_count <= 0 or prompt_tokens <= 0 or completion_tokens <= 0:
        raise ValueError("expected dimensions must be positive")
    bench_jsonl = bench_jsonl.resolve()
    trace_dir = trace_dir.resolve()
    record = _load_bench_record(bench_jsonl)
    requests = record.get("trace_requests")
    if not isinstance(requests, list) or len(requests) != expected_count:
        raise ValueError(
            f"expected {expected_count} indexed trace requests, found "
            f"{None if not isinstance(requests, list) else len(requests)}"
        )

    indexed_ids: list[str] = []
    manifest_entries: list[dict[str, Any]] = []
    common_identity: dict[str, Any] | None = None
    expected_positions = np.arange(
        prompt_tokens - 1,
        prompt_tokens + completion_tokens - 1,
        dtype=np.int32,
    )

    for expected_index, request in enumerate(requests):
        if request.get("request_index") != expected_index:
            raise ValueError(f"trace request index mismatch at {expected_index}")
        if request.get("success") is not True:
            raise ValueError(f"trace request {expected_index} was not successful")
        if request.get("prompt_kind") != "token_ids_int32_le":
            raise ValueError(f"trace request {expected_index} did not use token IDs")
        if request.get("prompt_tokens") != prompt_tokens:
            raise ValueError(f"trace request {expected_index} prompt length mismatch")
        if request.get("reported_prompt_tokens") != prompt_tokens:
            raise ValueError(
                f"trace request {expected_index} reported prompt length mismatch"
            )
        if request.get("completion_tokens") != completion_tokens:
            raise ValueError(
                f"trace request {expected_index} completion length mismatch"
            )
        trace_id = request.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError(f"trace request {expected_index} has no trace ID")
        if trace_id in indexed_ids:
            raise ValueError(f"duplicate trace ID: {trace_id}")
        indexed_ids.append(trace_id)

        json_path = trace_dir / f"{trace_id}.trace.json"
        npz_path = trace_dir / f"{trace_id}.trace.npz"
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if metadata.get("format") != TRACE_FORMAT:
            raise ValueError(f"unexpected trace format for {trace_id}")
        if metadata.get("trace_id") != trace_id or metadata.get("status") != "ok":
            raise ValueError(f"trace metadata status/identity mismatch for {trace_id}")
        request_metadata = metadata.get("request", {})
        if request_metadata.get("prompt_tokens") != prompt_tokens:
            raise ValueError(f"trace prompt length mismatch for {trace_id}")
        if request_metadata.get("completion_tokens") != completion_tokens:
            raise ValueError(f"trace completion length mismatch for {trace_id}")
        if request_metadata.get("completed_rows") != completion_tokens:
            raise ValueError(f"trace completed row mismatch for {trace_id}")

        identity = metadata.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"trace identity missing for {trace_id}")
        if identity.get("device", {}).get("uuid") != gpu_uuid:
            raise ValueError(f"trace GPU UUID mismatch for {trace_id}")
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError(f"model/pack/device identity changed at {trace_id}")

        artifact = metadata.get("artifact", {})
        npz_sha256 = _sha256(npz_path)
        if artifact.get("file") != npz_path.name:
            raise ValueError(f"trace artifact filename mismatch for {trace_id}")
        if artifact.get("size") != npz_path.stat().st_size:
            raise ValueError(f"trace artifact size mismatch for {trace_id}")
        if artifact.get("sha256") != npz_sha256:
            raise ValueError(f"trace artifact SHA-256 mismatch for {trace_id}")

        with np.load(npz_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(EXPECTED_ARRAYS):
                raise ValueError(f"trace array set mismatch for {trace_id}")
            values = {name: arrays[name] for name in arrays.files}
        for name, dtype in EXPECTED_ARRAYS.items():
            if values[name].dtype != dtype:
                raise ValueError(f"trace {name} dtype mismatch for {trace_id}")
        if values["expert_ids"].shape != (completion_tokens, 40, 8):
            raise ValueError(f"trace expert_ids shape mismatch for {trace_id}")
        for name in EXPECTED_ARRAYS.keys() - {"expert_ids"}:
            if values[name].shape != (completion_tokens,):
                raise ValueError(f"trace {name} shape mismatch for {trace_id}")
        if values["expert_ids"].size and (
            int(values["expert_ids"].min()) < 0 or int(values["expert_ids"].max()) > 255
        ):
            raise ValueError(f"trace expert ID range mismatch for {trace_id}")
        if values["phase"][0] != 0 or np.any(values["phase"][1:] != 1):
            raise ValueError(f"trace phase sequence mismatch for {trace_id}")
        if not np.array_equal(values["model_input_positions"], expected_positions):
            raise ValueError(f"trace model input positions mismatch for {trace_id}")
        if not np.array_equal(
            values["model_input_token_ids"][1:],
            values["sampled_token_ids"][:-1],
        ):
            raise ValueError(f"trace token attribution mismatch for {trace_id}")

        manifest_entries.append(
            {
                "request_index": expected_index,
                "trace_id": trace_id,
                "prompt_sha256": request.get("prompt_sha256"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "started_at": metadata.get("started_at"),
                "finished_at": metadata.get("finished_at"),
                "trace_json": json_path.name,
                "trace_json_sha256": _sha256(json_path),
                "trace_npz": npz_path.name,
                "trace_npz_size": npz_path.stat().st_size,
                "trace_npz_sha256": npz_sha256,
            }
        )

    expected_json = {f"{trace_id}.trace.json" for trace_id in indexed_ids}
    expected_npz = {f"{trace_id}.trace.npz" for trace_id in indexed_ids}
    actual_json = {path.name for path in trace_dir.glob("*.trace.json")}
    actual_npz = {path.name for path in trace_dir.glob("*.trace.npz")}
    partials = list(trace_dir.glob("*.partial"))
    if actual_json != expected_json or actual_npz != expected_npz:
        raise ValueError("trace directory contains missing or unindexed artifacts")
    if partials:
        raise ValueError(f"trace directory contains partial artifacts: {partials}")

    manifest_data = b"".join(
        (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8")
        for entry in manifest_entries
    )
    _atomic_write(output_manifest, manifest_data)
    manifest_sha256 = _sha256(output_manifest)
    summary: dict[str, Any] = {
        "format": DATASET_FORMAT,
        "request_count": expected_count,
        "prompt_tokens_per_request": prompt_tokens,
        "completion_tokens_per_request": completion_tokens,
        "trace_shape_per_request": [completion_tokens, 40, 8],
        "logical_dataset_shape": [expected_count, completion_tokens, 40, 8],
        "total_prompt_tokens": expected_count * prompt_tokens,
        "total_completion_tokens": expected_count * completion_tokens,
        "total_expert_selections": expected_count * completion_tokens * 40 * 8,
        "gpu_uuid": gpu_uuid,
        "identity": common_identity,
        "benchmark": {
            "file": str(bench_jsonl),
            "sha256": _sha256(bench_jsonl),
            "duration_seconds": record.get("benchmark_duration"),
            "output_token_throughput": record.get("output_token_throughput"),
        },
        "trace_manifest": {
            "file": output_manifest.name,
            "size": output_manifest.stat().st_size,
            "sha256": manifest_sha256,
        },
        "source_dataset": (
            None
            if source_dataset is None
            else {
                "file": str(source_dataset.resolve()),
                "size": source_dataset.stat().st_size,
                "sha256": _sha256(source_dataset.resolve()),
            }
        ),
    }
    summary_data = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write(output_summary, summary_data)
    _fsync_directory(output_manifest.parent)
    if output_summary.parent != output_manifest.parent:
        _fsync_directory(output_summary.parent)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-jsonl", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--completion-tokens", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--source-dataset", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_dataset(
            bench_jsonl=args.bench_jsonl,
            trace_dir=args.trace_dir,
            output_manifest=args.output_manifest,
            output_summary=args.output_summary,
            expected_count=args.expected_count,
            prompt_tokens=args.prompt_tokens,
            completion_tokens=args.completion_tokens,
            gpu_uuid=args.gpu_uuid,
            source_dataset=args.source_dataset,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
