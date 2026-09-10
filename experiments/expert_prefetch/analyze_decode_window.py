#!/usr/bin/env python3
"""Summarize resident decode windows and pinned H2D prefetch capacity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ANALYSIS_FORMAT = "SGLANG-QWEN35-PREFETCH-WINDOW-ANALYSIS-v1"
LAYER_METRICS = (
    "pre_router_ms",
    "router_ms",
    "shared_gap_ms",
    "routed_tail_ms",
    "layer_total_ms",
)
WINDOW_METRICS = (
    "route_to_next_router_start_ms",
    "route_to_next_router_ready_ms",
    "route_to_next_routed_expert_start_ms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {path}")
    try:
        with partial.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _stats(values: Iterable[float], prefix: str = "") -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("timing samples must be finite nonnegative values")
    percentiles = np.percentile(array, [50, 90, 95, 99])
    return {
        f"{prefix}mean_ms": float(array.mean()),
        f"{prefix}p50_ms": float(percentiles[0]),
        f"{prefix}p90_ms": float(percentiles[1]),
        f"{prefix}p95_ms": float(percentiles[2]),
        f"{prefix}p99_ms": float(percentiles[3]),
    }


def _load_decode_rows(
    path: Path, expected_steps: int, layer_ids: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_count = expected_steps * len(layer_ids)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} timing rows, got {len(rows)}")
    for index, row in enumerate(rows):
        decode_index, layer_offset = divmod(index, len(layer_ids))
        if row.get("decode_index") != decode_index:
            raise ValueError("decode timing request axis is not contiguous")
        if row.get("step_id") != decode_index + 1:
            raise ValueError("decode timing step IDs are not contiguous")
        if row.get("layer_id") != layer_ids[layer_offset]:
            raise ValueError("decode timing layer axis is not contiguous")
        boundaries = [
            float(row[f"{name}_ms"])
            for name in (
                "layer_start",
                "router_start",
                "router_ready",
                "routed_expert_start",
                "layer_end",
            )
        ]
        if any(not np.isfinite(value) or value < 0.0 for value in boundaries):
            raise ValueError("decode timing contains an invalid boundary")
        if any(left > right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("decode timing boundaries are not monotonic")
    return rows


def _layer_summary(
    rows: list[dict[str, Any]], layer_ids: tuple[int, ...]
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer_id"])].append(row)
    result = []
    for layer_id in layer_ids:
        samples = grouped[layer_id]
        row: dict[str, Any] = {
            "layer_id": layer_id,
            "layer_type": samples[0]["layer_type"],
            "routed_format": samples[0]["routed_format"],
            "samples": len(samples),
        }
        for metric in LAYER_METRICS:
            row.update(
                _stats((float(sample[metric]) for sample in samples), metric[:-2])
            )
        result.append(row)
    return result


def _window_samples(
    rows: list[dict[str, Any]], expected_steps: int, layer_ids: tuple[int, ...]
) -> dict[tuple[int, int], dict[str, list[float]]]:
    width = len(layer_ids)
    windows = {
        (source, target): {metric: [] for metric in WINDOW_METRICS}
        for source, target in zip(layer_ids, layer_ids[1:])
    }
    for decode_index in range(expected_steps):
        step = rows[decode_index * width : (decode_index + 1) * width]
        for current, following in zip(step, step[1:]):
            source = int(current["layer_id"])
            target = int(following["layer_id"])
            router_ready = float(current["router_ready_ms"])
            values = {
                "route_to_next_router_start_ms": (
                    float(following["router_start_ms"]) - router_ready
                ),
                "route_to_next_router_ready_ms": (
                    float(following["router_ready_ms"]) - router_ready
                ),
                "route_to_next_routed_expert_start_ms": (
                    float(following["routed_expert_start_ms"]) - router_ready
                ),
            }
            if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
                raise ValueError("adjacent decode timing windows are invalid")
            for metric, value in values.items():
                windows[(source, target)][metric].append(value)
    return windows


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    layer_ids = tuple(range(args.layer_start, args.layer_end))
    if len(layer_ids) < 2:
        raise ValueError("analysis requires at least two consecutive layers")
    decode_rows = _load_decode_rows(
        args.decode_jsonl.resolve(), args.expected_steps, layer_ids
    )
    pp_summary = json.loads(args.pp_summary.read_text(encoding="utf-8"))
    if pp_summary.get("measurement_decode_steps") != args.expected_steps:
        raise ValueError("PP summary decode step count mismatch")
    if pp_summary.get("back_layer_ids") != list(layer_ids):
        raise ValueError("PP summary layer slice mismatch")
    token_checks = pp_summary.get("token_checks", {})
    if token_checks != {
        "first_oracle_tokens_match_exp0001": True,
        "timed_equals_control": True,
    }:
        raise ValueError("PP summary did not pass token correctness checks")

    layer_rows = _layer_summary(decode_rows, layer_ids)
    window_values = _window_samples(decode_rows, args.expected_steps, layer_ids)
    window_rows: list[dict[str, Any]] = []
    for source, target in zip(layer_ids, layer_ids[1:]):
        values = window_values[(source, target)]
        row: dict[str, Any] = {
            "source_layer": source,
            "target_layer": target,
            "target_layer_type": ("full_attention" if (target + 1) % 4 == 0 else "gdn"),
            "target_routed_format": "fp16" if target == 39 else "nvfp4",
            "offload_relevant": target <= 38,
            "samples": args.expected_steps,
        }
        for metric in WINDOW_METRICS:
            row.update(_stats(values[metric], metric[:-2]))
        window_rows.append(row)

    with np.load(args.h2d_npz, allow_pickle=False) as data:
        required = {
            "format",
            "candidate_counts",
            "transfer_bytes",
            "latency_ms",
            "payload_size",
            "warmup",
            "samples",
            "source_payload_sha256",
        }
        if set(data.files) != required:
            raise ValueError("H2D NPZ has an unexpected array set")
        candidates = data["candidate_counts"].copy()
        transfer_bytes = data["transfer_bytes"].copy()
        latency_ms = data["latency_ms"].copy()
        payload_size = int(data["payload_size"])
        raw_h2d_format = str(data["format"])
    if (
        candidates.dtype != np.int32
        or transfer_bytes.dtype != np.int64
        or latency_ms.dtype != np.float64
        or latency_ms.shape != (len(candidates), args.expected_h2d_samples)
        or not np.array_equal(
            transfer_bytes, candidates.astype(np.int64) * payload_size
        )
        or not np.all(np.isfinite(latency_ms))
        or np.any(latency_ms <= 0.0)
    ):
        raise ValueError("H2D sample arrays violate their contract")

    h2d_rows: list[dict[str, Any]] = []
    h2d_p95: dict[int, float] = {}
    for index, candidate in enumerate(candidates.tolist()):
        stats = _stats(latency_ms[index])
        h2d_p95[int(candidate)] = stats["p95_ms"]
        h2d_rows.append(
            {
                "candidate_count": int(candidate),
                "transfer_bytes": int(transfer_bytes[index]),
                "samples": args.expected_h2d_samples,
                **stats,
                "effective_gib_per_second_at_p50": (
                    int(transfer_bytes[index]) / stats["p50_ms"] * 1000.0 / (1 << 30)
                ),
            }
        )

    capacity_rows: list[dict[str, Any]] = []
    max_k_by_transition: dict[str, int] = {}
    overall_deadlines: list[float] = []
    for source, target in zip(layer_ids, layer_ids[1:]):
        key = f"{source}->{target}"
        deadlines = np.asarray(
            window_values[(source, target)]["route_to_next_routed_expert_start_ms"],
            dtype=np.float64,
        )
        relevant = target <= 38
        if relevant:
            overall_deadlines.extend(deadlines.tolist())
        transition_max = 0
        local_rows = []
        for candidate in candidates.tolist():
            threshold = h2d_p95[int(candidate)]
            fit_count = int(np.count_nonzero(deadlines >= threshold))
            fit_rate = fit_count / len(deadlines)
            if fit_rate >= 0.95:
                transition_max = int(candidate)
            local_rows.append(
                {
                    "source_layer": source,
                    "target_layer": target,
                    "offload_relevant": relevant,
                    "candidate_count": int(candidate),
                    "transfer_bytes": int(candidate) * payload_size,
                    "h2d_p95_ms": threshold,
                    "window_samples": len(deadlines),
                    "fit_count": fit_count,
                    "fit_rate": fit_rate,
                    "meets_95_percent": fit_rate >= 0.95,
                }
            )
        max_k_by_transition[key] = transition_max
        for row in local_rows:
            row["transition_max_k_95"] = transition_max
            capacity_rows.append(row)

    overall_array = np.asarray(overall_deadlines, dtype=np.float64)
    overall_fit = {}
    overall_max_k = 0
    for candidate in candidates.tolist():
        rate = float(np.mean(overall_array >= h2d_p95[int(candidate)]))
        overall_fit[str(candidate)] = rate
        if rate >= 0.95:
            overall_max_k = int(candidate)

    layer_fields = ["layer_id", "layer_type", "routed_format", "samples"] + [
        f"{metric[:-2]}{stat}_ms"
        for metric in LAYER_METRICS
        for stat in ("mean", "p50", "p90", "p95", "p99")
    ]
    window_fields = [
        "source_layer",
        "target_layer",
        "target_layer_type",
        "target_routed_format",
        "offload_relevant",
        "samples",
    ] + [
        f"{metric[:-2]}{stat}_ms"
        for metric in WINDOW_METRICS
        for stat in ("mean", "p50", "p90", "p95", "p99")
    ]
    h2d_fields = [
        "candidate_count",
        "transfer_bytes",
        "samples",
        "mean_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "effective_gib_per_second_at_p50",
    ]
    capacity_fields = [
        "source_layer",
        "target_layer",
        "offload_relevant",
        "candidate_count",
        "transfer_bytes",
        "h2d_p95_ms",
        "window_samples",
        "fit_count",
        "fit_rate",
        "meets_95_percent",
        "transition_max_k_95",
    ]
    outputs = {
        args.layer_summary: _csv_bytes(layer_rows, layer_fields),
        args.window_summary: _csv_bytes(window_rows, window_fields),
        args.h2d_summary: _csv_bytes(h2d_rows, h2d_fields),
        args.capacity_summary: _csv_bytes(capacity_rows, capacity_fields),
    }
    summary: dict[str, Any] = {
        "format": ANALYSIS_FORMAT,
        "interpretation": (
            "H2D-only upper bound; excludes predictor, route D2H, SSD, checksum, "
            "CPU scheduling, and copy/compute contention"
        ),
        "decode_steps": args.expected_steps,
        "v100_layer_ids": list(layer_ids),
        "layer_samples": len(decode_rows),
        "adjacent_window_samples": args.expected_steps * (len(layer_ids) - 1),
        "offload_relevant_transitions": [
            f"{source}->{target}"
            for source, target in zip(layer_ids, layer_ids[1:])
            if target <= 38
        ],
        "offload_relevant_window_samples": len(overall_deadlines),
        "candidate_counts": candidates.tolist(),
        "payload_size": payload_size,
        "h2d_format": raw_h2d_format,
        "overall_fit_rate_by_candidate": overall_fit,
        "overall_max_k_95": overall_max_k,
        "max_k_95_by_transition": max_k_by_transition,
        "inputs": {
            "decode_jsonl": {
                "file": str(args.decode_jsonl.resolve()),
                "sha256": _sha256(args.decode_jsonl.resolve()),
            },
            "pp_summary": {
                "file": str(args.pp_summary.resolve()),
                "sha256": _sha256(args.pp_summary.resolve()),
            },
            "h2d_npz": {
                "file": str(args.h2d_npz.resolve()),
                "sha256": _sha256(args.h2d_npz.resolve()),
            },
        },
        "outputs": {
            str(path.resolve()): hashlib.sha256(data).hexdigest()
            for path, data in outputs.items()
        },
    }
    for path, data in outputs.items():
        _atomic_write(path, data)
    summary_data = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write(args.experiment_summary, summary_data)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-jsonl", type=Path, required=True)
    parser.add_argument("--pp-summary", type=Path, required=True)
    parser.add_argument("--h2d-npz", type=Path, required=True)
    parser.add_argument("--layer-summary", type=Path, required=True)
    parser.add_argument("--window-summary", type=Path, required=True)
    parser.add_argument("--h2d-summary", type=Path, required=True)
    parser.add_argument("--capacity-summary", type=Path, required=True)
    parser.add_argument("--experiment-summary", type=Path, required=True)
    parser.add_argument("--layer-start", type=int, default=17)
    parser.add_argument("--layer-end", type=int, default=40)
    parser.add_argument("--expected-steps", type=int, default=256)
    parser.add_argument("--expected-h2d-samples", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = analyze(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
