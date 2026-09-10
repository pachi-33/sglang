#!/usr/bin/env python3
"""Evaluate cross-request unions of frozen route predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NUM_EXPERTS = 256
NUM_LAYERS = 40
TRUE_TOP_K = 8
FIRST_TARGET_ROW = 1
FORMAT = "SGLANG-BATCH-UNION-ROUTE-PREDICTION-EVAL-v1"
SUBSET_FORMAT = "SGLANG-BATCH-UNION-REQUEST-SUBSETS-v1"
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with partial.open("xb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with partial.open("xb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def make_bitsets(expert_ids: np.ndarray) -> np.ndarray:
    if expert_ids.dtype != np.uint8 or expert_ids.ndim != 4:
        raise ValueError(
            "expert IDs must be uint8 [request, position, layer, rank]"
        )
    bitsets = np.zeros(expert_ids.shape[:-1] + (4,), dtype=np.uint64)
    for rank in range(expert_ids.shape[-1]):
        ids = expert_ids[..., rank].astype(np.uint64)
        words = ids // np.uint64(64)
        masks = np.left_shift(np.uint64(1), ids % np.uint64(64))
        for word in range(4):
            bitsets[..., word] |= np.where(
                words == word, masks, np.uint64(0)
            )
    return bitsets


def count_bits(bitsets: np.ndarray) -> np.ndarray:
    byte_view = bitsets.view(np.uint8).reshape(
        bitsets.shape[:-1] + (bitsets.shape[-1] * 8,)
    )
    return POPCOUNT[byte_view].sum(axis=-1).astype(np.int16)


def assert_unique_last_axis(name: str, values: np.ndarray) -> None:
    sorted_values = np.sort(values, axis=-1).astype(np.int16)
    if np.any(np.diff(sorted_values, axis=-1) == 0):
        raise ValueError(f"{name} contains duplicate expert IDs")


@dataclass
class ScopeAccumulator:
    cell_count: int = 0
    subset_count: int = 0
    sum_m: int = 0
    sum_n: int = 0
    sum_h: int = 0
    sum_m_over_n: float = 0.0
    sum_h_over_m: float = 0.0
    sum_h_over_n: float = 0.0
    fully_covered: int = 0
    subset_mean_m_over_n: list[np.ndarray] = field(default_factory=list)
    subset_mean_recall: list[np.ndarray] = field(default_factory=list)

    def update(self, m: np.ndarray, n: np.ndarray, h: np.ndarray) -> None:
        if m.shape != n.shape or m.shape != h.shape or m.ndim != 3:
            raise ValueError("scope arrays must share [subset, position, layer]")
        if np.any(m <= 0) or np.any(n <= 0):
            raise ValueError("union cardinalities must be positive")
        m_over_n = m / n
        h_over_m = h / m
        h_over_n = h / n
        self.cell_count += m.size
        self.subset_count += m.shape[0]
        self.sum_m += int(m.sum())
        self.sum_n += int(n.sum())
        self.sum_h += int(h.sum())
        self.sum_m_over_n += float(m_over_n.sum())
        self.sum_h_over_m += float(h_over_m.sum())
        self.sum_h_over_n += float(h_over_n.sum())
        self.fully_covered += int(np.count_nonzero(h == m))
        self.subset_mean_m_over_n.append(m_over_n.mean(axis=(1, 2)))
        self.subset_mean_recall.append(h_over_m.mean(axis=(1, 2)))

    def finalize(self) -> dict[str, Any]:
        if not self.cell_count or not self.subset_count:
            raise ValueError("cannot finalize empty accumulator")
        ratio_samples = np.concatenate(self.subset_mean_m_over_n)
        recall_samples = np.concatenate(self.subset_mean_recall)
        if ratio_samples.size != self.subset_count:
            raise AssertionError("subset statistics count mismatch")
        return {
            "subset_count": self.subset_count,
            "token_layer_cells": self.cell_count,
            "mean_actual_union_m": self.sum_m / self.cell_count,
            "mean_predicted_union_n": self.sum_n / self.cell_count,
            "mean_intersection_h": self.sum_h / self.cell_count,
            "mean_m_over_n": self.sum_m_over_n / self.cell_count,
            "ratio_of_sums_m_over_n": self.sum_m / self.sum_n,
            "mean_batch_recall_h_over_m": self.sum_h_over_m / self.cell_count,
            "global_batch_recall_h_over_m": self.sum_h / self.sum_m,
            "mean_batch_precision_h_over_n": self.sum_h_over_n
            / self.cell_count,
            "global_batch_precision_h_over_n": self.sum_h / self.sum_n,
            "mean_missed_actual": (self.sum_m - self.sum_h)
            / self.cell_count,
            "mean_wasted_predicted": (self.sum_n - self.sum_h)
            / self.cell_count,
            "fully_covered_rate": self.fully_covered / self.cell_count,
            "subset_mean_m_over_n_p05": float(
                np.quantile(ratio_samples, 0.05)
            ),
            "subset_mean_m_over_n_p50": float(
                np.quantile(ratio_samples, 0.50)
            ),
            "subset_mean_m_over_n_p95": float(
                np.quantile(ratio_samples, 0.95)
            ),
            "subset_mean_recall_p05": float(
                np.quantile(recall_samples, 0.05)
            ),
            "subset_mean_recall_p50": float(
                np.quantile(recall_samples, 0.50)
            ),
            "subset_mean_recall_p95": float(
                np.quantile(recall_samples, 0.95)
            ),
        }


@dataclass
class LayerAccumulator:
    cells_per_layer: int = 0
    sum_m: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LAYERS, dtype=np.int64)
    )
    sum_n: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LAYERS, dtype=np.int64)
    )
    sum_h: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LAYERS, dtype=np.int64)
    )
    sum_m_over_n: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LAYERS, dtype=np.float64)
    )
    sum_h_over_m: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LAYERS, dtype=np.float64)
    )
    sum_h_over_n: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LAYERS, dtype=np.float64)
    )
    fully_covered: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_LAYERS, dtype=np.int64)
    )

    def update(self, m: np.ndarray, n: np.ndarray, h: np.ndarray) -> None:
        self.cells_per_layer += m.shape[0] * m.shape[1]
        self.sum_m += m.sum(axis=(0, 1), dtype=np.int64)
        self.sum_n += n.sum(axis=(0, 1), dtype=np.int64)
        self.sum_h += h.sum(axis=(0, 1), dtype=np.int64)
        self.sum_m_over_n += (m / n).sum(axis=(0, 1))
        self.sum_h_over_m += (h / m).sum(axis=(0, 1))
        self.sum_h_over_n += (h / n).sum(axis=(0, 1))
        self.fully_covered += (h == m).sum(axis=(0, 1), dtype=np.int64)

    def finalize(self, *, batch_size: int, candidate_count: int) -> list[dict[str, Any]]:
        rows = []
        for layer in range(NUM_LAYERS):
            rows.append(
                {
                    "batch_size_b": batch_size,
                    "candidate_count_k": candidate_count,
                    "layer_id": layer,
                    "cells": self.cells_per_layer,
                    "mean_actual_union_m": self.sum_m[layer]
                    / self.cells_per_layer,
                    "mean_predicted_union_n": self.sum_n[layer]
                    / self.cells_per_layer,
                    "mean_m_over_n": self.sum_m_over_n[layer]
                    / self.cells_per_layer,
                    "ratio_of_sums_m_over_n": self.sum_m[layer]
                    / self.sum_n[layer],
                    "mean_batch_recall_h_over_m": self.sum_h_over_m[layer]
                    / self.cells_per_layer,
                    "global_batch_recall_h_over_m": self.sum_h[layer]
                    / self.sum_m[layer],
                    "mean_batch_precision_h_over_n": self.sum_h_over_n[layer]
                    / self.cells_per_layer,
                    "global_batch_precision_h_over_n": self.sum_h[layer]
                    / self.sum_n[layer],
                    "mean_missed_actual": (
                        self.sum_m[layer] - self.sum_h[layer]
                    )
                    / self.cells_per_layer,
                    "mean_wasted_predicted": (
                        self.sum_n[layer] - self.sum_h[layer]
                    )
                    / self.cells_per_layer,
                    "fully_covered_rate": self.fully_covered[layer]
                    / self.cells_per_layer,
                }
            )
        return rows


def enumerate_subsets(request_count: int, batch_size: int) -> np.ndarray:
    count = math.comb(request_count, batch_size)
    if count > 20_000:
        raise ValueError(
            f"C({request_count}, {batch_size})={count} exceeds exact limit"
        )
    return np.asarray(
        list(itertools.combinations(range(request_count), batch_size)),
        dtype=np.int32,
    ).reshape(count, batch_size)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.test_npz, allow_pickle=False) as archive:
        routes = archive["expert_ids"].copy()
        request_indices = archive["request_indices"].copy()
        trace_ids = archive["trace_ids"].copy()
        split_name = str(archive["split_name"])
    with np.load(args.predictions_npz, allow_pickle=False) as archive:
        predictions = archive["test_candidates"].copy()
        predicted_request_indices = archive["test_request_indices"].copy()
        predicted_trace_ids = archive["test_trace_ids"].copy()
        checkpoint_sha256 = str(archive["checkpoint_sha256"])
        artifact_candidate_counts = archive["candidate_counts"].copy()

    if split_name != "test":
        raise ValueError(f"expected test split, got {split_name!r}")
    if routes.shape != (16, 256, NUM_LAYERS, TRUE_TOP_K):
        raise ValueError(f"unexpected test routes shape: {routes.shape}")
    if predictions.shape != (16, 255, NUM_LAYERS, 32):
        raise ValueError(f"unexpected prediction shape: {predictions.shape}")
    if routes.dtype != np.uint8 or predictions.dtype != np.uint8:
        raise ValueError("route and prediction IDs must use uint8")
    if not np.array_equal(request_indices, predicted_request_indices):
        raise ValueError("prediction/test request indices do not align")
    if not np.array_equal(trace_ids, predicted_trace_ids):
        raise ValueError("prediction/test trace IDs do not align")
    if list(artifact_candidate_counts) != args.candidate_counts:
        raise ValueError("prediction artifact candidate counts changed")
    assert_unique_last_axis("actual Top-8", routes)
    assert_unique_last_axis("predicted Top-32", predictions)

    targets = routes[:, FIRST_TARGET_ROW:, :, :]
    actual_bits = make_bitsets(targets)
    predicted_bits = {
        k: make_bitsets(predictions[..., :k]) for k in args.candidate_counts
    }
    position_slices = {
        "all": slice(None),
        "cold_start": slice(0, args.history_tokens - 1),
        "steady_state": slice(args.history_tokens - 1, None),
    }
    layer_slices = {"all_layers": slice(None), "offload_layers_1_38": slice(1, 39)}

    subset_arrays: dict[str, np.ndarray] = {
        "format": np.asarray(SUBSET_FORMAT),
        "request_indices": request_indices,
        "trace_ids": trace_ids,
    }
    metric_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    nested_results: dict[str, Any] = {}

    for batch_size in args.batch_sizes:
        subsets = enumerate_subsets(routes.shape[0], batch_size)
        subset_arrays[f"b{batch_size}_local_indices"] = subsets
        subset_arrays[f"b{batch_size}_request_indices"] = request_indices[subsets]
        accumulators = {
            (k, position_scope, layer_scope): ScopeAccumulator()
            for k in args.candidate_counts
            for position_scope in position_slices
            for layer_scope in layer_slices
        }
        layer_accumulators = {
            k: LayerAccumulator() for k in args.candidate_counts
        }

        for start in range(0, len(subsets), args.subset_chunk_size):
            subset_chunk = subsets[start : start + args.subset_chunk_size]
            actual_union = np.bitwise_or.reduce(
                actual_bits[subset_chunk], axis=1
            )
            m = count_bits(actual_union)
            if np.any(m > min(NUM_EXPERTS, TRUE_TOP_K * batch_size)):
                raise AssertionError("actual union exceeds cardinality bound")
            for candidate_count in args.candidate_counts:
                predicted_union = np.bitwise_or.reduce(
                    predicted_bits[candidate_count][subset_chunk], axis=1
                )
                n = count_bits(predicted_union)
                h = count_bits(actual_union & predicted_union)
                if np.any(n > min(NUM_EXPERTS, candidate_count * batch_size)):
                    raise AssertionError("predicted union exceeds cardinality bound")
                if np.any(h > m) or np.any(h > n):
                    raise AssertionError("intersection exceeds union")
                layer_accumulators[candidate_count].update(m, n, h)
                for position_scope, position_slice in position_slices.items():
                    for layer_scope, layer_slice in layer_slices.items():
                        accumulators[
                            (candidate_count, position_scope, layer_scope)
                        ].update(
                            m[:, position_slice, layer_slice],
                            n[:, position_slice, layer_slice],
                            h[:, position_slice, layer_slice],
                        )

        batch_result: dict[str, Any] = {
            "sampling": "exact-uniform-subset-enumeration",
            "subset_count": len(subsets),
            "candidate_counts": {},
        }
        for candidate_count in args.candidate_counts:
            candidate_result: dict[str, Any] = {}
            for position_scope in position_slices:
                candidate_result[position_scope] = {}
                for layer_scope in layer_slices:
                    values = accumulators[
                        (candidate_count, position_scope, layer_scope)
                    ].finalize()
                    candidate_result[position_scope][layer_scope] = values
                    metric_rows.append(
                        {
                            "batch_size_b": batch_size,
                            "candidate_count_k": candidate_count,
                            "position_scope": position_scope,
                            "layer_scope": layer_scope,
                            **values,
                        }
                    )
            batch_result["candidate_counts"][str(candidate_count)] = candidate_result
            layer_rows.extend(
                layer_accumulators[candidate_count].finalize(
                    batch_size=batch_size,
                    candidate_count=candidate_count,
                )
            )
        nested_results[str(batch_size)] = batch_result

    # Structural invariants across K and regression to EXP-0003 for B=1.
    expected_single_request_recall = {
        8: 0.3434168198529412,
        16: 0.5020833333333333,
        24: 0.5993589154411765,
        32: 0.6685401348039216,
    }
    for batch_size in args.batch_sizes:
        previous_n = None
        previous_h = None
        previous_ratio = None
        actual_m = None
        for candidate_count in args.candidate_counts:
            values = nested_results[str(batch_size)]["candidate_counts"][
                str(candidate_count)
            ]["all"]["all_layers"]
            if actual_m is None:
                actual_m = values["mean_actual_union_m"]
            elif not np.isclose(actual_m, values["mean_actual_union_m"]):
                raise AssertionError("actual M changed across K")
            current_n = values["mean_predicted_union_n"]
            current_h = values["mean_intersection_h"]
            current_ratio = values["mean_m_over_n"]
            if previous_n is not None and current_n + 1e-12 < previous_n:
                raise AssertionError("N decreased as K increased")
            if previous_h is not None and current_h + 1e-12 < previous_h:
                raise AssertionError("H decreased as K increased")
            if previous_ratio is not None and current_ratio > previous_ratio + 1e-12:
                raise AssertionError("M/N increased as K increased")
            previous_n, previous_h, previous_ratio = (
                current_n,
                current_h,
                current_ratio,
            )
            if batch_size == 1:
                if not np.isclose(values["mean_actual_union_m"], TRUE_TOP_K):
                    raise AssertionError("B=1 actual M is not 8")
                if not np.isclose(values["mean_predicted_union_n"], candidate_count):
                    raise AssertionError("B=1 predicted N is not K")
                if not np.isclose(
                    values["global_batch_recall_h_over_m"],
                    expected_single_request_recall[candidate_count],
                    rtol=0,
                    atol=1e-15,
                ):
                    raise AssertionError(
                        f"B=1 Recall@{candidate_count} did not reproduce EXP-0003"
                    )

    subset_path = args.output_artifacts / "evaluated_batch_subsets.npz"
    atomic_npz(subset_path, subset_arrays)
    source = {
        "experiment": "EXP-0003",
        "predictions": {
            "path": str(args.predictions_npz.resolve()),
            "size": args.predictions_npz.stat().st_size,
            "sha256": sha256(args.predictions_npz),
        },
        "test_split": {
            "path": str(args.test_npz.resolve()),
            "size": args.test_npz.stat().st_size,
            "sha256": sha256(args.test_npz),
        },
        "checkpoint_sha256": checkpoint_sha256,
    }
    detailed: dict[str, Any] = {
        "format": FORMAT,
        "source": source,
        "parameters": {
            "split": "test",
            "request_count": 16,
            "batch_sizes_b": args.batch_sizes,
            "candidate_counts_k": args.candidate_counts,
            "target_rows": [1, 255],
            "history_tokens": args.history_tokens,
            "subset_sampling": "exact enumeration without replacement",
        },
        "semantics": {
            "m": "size of actual Top-8 union across B requests at one position/layer",
            "n": "size of predicted Top-K union across B requests at one position/layer",
            "h": "size of intersection between the actual and predicted unions",
            "primary_user_metric": "macro mean of M/N over subset, position, layer cells",
            "batch_recall": "H/M; this, rather than M/N, measures actual coverage",
        },
        "results": nested_results,
    }
    detailed_path = args.output_results / "batch_union_metrics.json"
    atomic_json(detailed_path, detailed)
    atomic_write(
        args.output_results / "batch_union_metrics.csv", csv_bytes(metric_rows)
    )
    atomic_write(
        args.output_results / "batch_union_by_layer.csv", csv_bytes(layer_rows)
    )

    primary_rows = [
        row
        for row in metric_rows
        if row["position_scope"] == "all"
        and row["layer_scope"] == "offload_layers_1_38"
    ]
    summary = {
        "format": "SGLANG-BATCH-UNION-ROUTE-PREDICTION-SUMMARY-v1",
        "experiment_id": "EXP-0005",
        "source": source,
        "primary_scope": {
            "split": "test",
            "target_rows": [1, 255],
            "layers": [1, 38],
            "aggregation": "exact mean over all B-request subsets, positions, and layers",
        },
        "primary_results": primary_rows,
        "artifacts": {
            "detailed_metrics": {
                "path": "results/batch_union_metrics.json",
                "size": detailed_path.stat().st_size,
                "sha256": sha256(detailed_path),
            },
            "evaluated_subsets": {
                "path": "artifacts/evaluated_batch_subsets.npz",
                "size": subset_path.stat().st_size,
                "sha256": sha256(subset_path),
            },
        },
    }
    atomic_json(args.output_results / "batch_union_summary.json", summary)
    fsync_directory(args.output_results)
    fsync_directory(args.output_artifacts)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--output-artifacts", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument(
        "--candidate-counts", type=int, nargs="+", default=[8, 16, 24, 32]
    )
    parser.add_argument("--history-tokens", type=int, default=8)
    parser.add_argument("--subset-chunk-size", type=int, default=16)
    args = parser.parse_args()
    args.batch_sizes = sorted(set(args.batch_sizes))
    args.candidate_counts = sorted(set(args.candidate_counts))
    if args.batch_sizes != [1, 2, 4, 8, 16]:
        parser.error("this experiment requires B = 1, 2, 4, 8, 16")
    if args.candidate_counts != [8, 16, 24, 32]:
        parser.error("this experiment requires K = 8, 16, 24, 32")
    if args.history_tokens != 8 or args.subset_chunk_size <= 0:
        parser.error("history-tokens must be 8 and chunk size must be positive")
    try:
        summary = evaluate(args)
    except (OSError, ValueError, AssertionError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
