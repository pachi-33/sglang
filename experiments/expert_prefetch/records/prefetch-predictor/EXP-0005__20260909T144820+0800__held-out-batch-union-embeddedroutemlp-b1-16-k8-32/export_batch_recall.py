#!/usr/bin/env python3
"""Export EXP-0005 metrics using the user-defined M/N recall notation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any


FORMAT = "SGLANG-BATCH-UNION-RECALL-SUMMARY-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with partial.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def export(source_path: Path, output_json: Path, output_csv: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = source.get("primary_results")
    if not isinstance(source_rows, list) or len(source_rows) != 20:
        raise ValueError("expected 20 B/K primary result rows")

    rows: list[dict[str, Any]] = []
    for value in source_rows:
        predicted_correct_m = value["mean_intersection_h"]
        actual_unique_n = value["mean_actual_union_m"]
        mean_m_over_n = value["mean_batch_recall_h_over_m"]
        ratio_of_sums = value["global_batch_recall_h_over_m"]
        if not 0 <= predicted_correct_m <= actual_unique_n:
            raise ValueError("predicted-correct M must be in [0, actual N]")
        if not 0 <= mean_m_over_n <= 1 or not 0 <= ratio_of_sums <= 1:
            raise ValueError("M/N recall must be in [0, 1]")
        rows.append(
            {
                "batch_size_b": value["batch_size_b"],
                "candidate_count_k": value["candidate_count_k"],
                "subset_count": value["subset_count"],
                "token_layer_cells": value["token_layer_cells"],
                "mean_predicted_correct_m": predicted_correct_m,
                "mean_actual_unique_n": actual_unique_n,
                "mean_m_over_n": mean_m_over_n,
                "ratio_of_sums_m_over_n": ratio_of_sums,
                "mean_predicted_union_size": value["mean_predicted_union_n"],
                "mean_missed_actual": value["mean_missed_actual"],
                "mean_wasted_predicted": value["mean_wasted_predicted"],
                "fully_covered_rate": value["fully_covered_rate"],
                "subset_mean_m_over_n_p05": value["subset_mean_recall_p05"],
                "subset_mean_m_over_n_p50": value["subset_mean_recall_p50"],
                "subset_mean_m_over_n_p95": value["subset_mean_recall_p95"],
            }
        )

    expected_pairs = [
        (batch_size, candidate_count)
        for batch_size in (1, 2, 4, 8, 16)
        for candidate_count in (8, 16, 24, 32)
    ]
    actual_pairs = [
        (row["batch_size_b"], row["candidate_count_k"]) for row in rows
    ]
    if actual_pairs != expected_pairs:
        raise ValueError(f"unexpected B/K ordering: {actual_pairs}")

    result = {
        "format": FORMAT,
        "experiment_id": "EXP-0005",
        "authoritative_metric": "mean_m_over_n",
        "definitions": {
            "actual_set": (
                "union of the B requests' true Top-8 experts at one output "
                "position and layer"
            ),
            "predicted_set": (
                "union of the B requests' predicted Top-K experts at the "
                "same output position and layer"
            ),
            "m": "size of actual_set intersection predicted_set",
            "n": "size of actual_set",
            "m_over_n": "batch union recall; predicted-correct experts / actual experts",
            "mean_m_over_n": (
                "macro mean of per-cell M/N over request subset, output "
                "position, and offload layer"
            ),
            "ratio_of_sums_m_over_n": "sum(M) / sum(N), reported separately",
        },
        "scope": {
            "split": "held-out test",
            "request_count": 16,
            "batch_sizes_b": [1, 2, 4, 8, 16],
            "candidate_counts_k": [8, 16, 24, 32],
            "output_rows": [1, 255],
            "layers": [1, 38],
            "subset_sampling": "exact enumeration without replacement",
            "total_request_subsets": 14827,
        },
        "source": {
            "file": str(source_path.resolve()),
            "size": source_path.stat().st_size,
            "sha256": sha256(source_path),
            "prediction_artifact_sha256": source["source"]["predictions"]["sha256"],
            "test_split_sha256": source["source"]["test_split"]["sha256"],
            "checkpoint_sha256": source["source"]["checkpoint_sha256"],
        },
        "results": rows,
    }
    atomic_write(
        output_json,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    atomic_write(output_csv, csv_bytes(rows))
    directory_fd = os.open(
        output_json.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export(args.source, args.output_json, args.output_csv)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
