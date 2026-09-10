#!/usr/bin/env python3
"""Measure deduplicated experts for random request subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

FORMAT = "SGLANG-EXPERT-PREFETCH-UNIQUE-EXPERTS-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    with partial.open("wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def make_bitsets(expert_ids: np.ndarray) -> np.ndarray:
    if expert_ids.dtype != np.uint8 or expert_ids.ndim != 4:
        raise ValueError(
            "expert_ids must be uint8 [request, position, layer, rank]"
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


def analyze(
    dataset_path: Path,
    output_dir: Path,
    k_values: list[int],
    trials: int,
    seed: int,
) -> dict[str, object]:
    with np.load(dataset_path, allow_pickle=False) as dataset:
        expert_ids = dataset["expert_ids"].copy()
        source_format = str(dataset["format"])
        manifest_sha256 = str(dataset["source_manifest_sha256"])

    request_count, positions, layers, top_k = expert_ids.shape
    if sorted(set(k_values)) != k_values:
        raise ValueError("k-values must be unique and sorted")
    if not k_values or k_values[0] < 1 or k_values[-1] > request_count:
        raise ValueError(f"k-values must be in 1..{request_count}")
    if trials < 1:
        raise ValueError("trials must be positive")

    bitsets = make_bitsets(expert_ids)
    request_layer_bits = np.bitwise_or.reduce(bitsets, axis=1)
    popcount = np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1
    ).sum(axis=1)

    def count_bits(value: np.ndarray) -> np.ndarray:
        return popcount[value.view(np.uint8)].sum(axis=-1)

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    for k in k_values:
        if k == 1:
            mode = "exact-all-singletons"
            subsets = [np.asarray([index]) for index in range(request_count)]
        elif k == request_count:
            mode = "exact-full-dataset"
            subsets = [np.arange(request_count)]
        else:
            mode = "monte-carlo-without-replacement"
            subsets = [
                rng.choice(request_count, size=k, replace=False)
                for _ in range(trials)
            ]

        whole_samples = []
        same_position_samples = []
        for subset in subsets:
            whole_union = np.bitwise_or.reduce(
                request_layer_bits[subset], axis=0
            )
            whole_samples.append(count_bits(whole_union))
            position_union = np.bitwise_or.reduce(bitsets[subset], axis=0)
            same_position_samples.append(
                count_bits(position_union).mean(axis=0)
            )

        whole = np.asarray(whole_samples, dtype=np.float64)
        same_position = np.asarray(same_position_samples, dtype=np.float64)
        whole_by_layer = whole.mean(axis=0)
        same_position_by_layer = same_position.mean(axis=0)
        row = {
            "k": k,
            "sampling_mode": mode,
            "subset_count": len(subsets),
            "whole_request_mean_unique_per_layer": round(
                float(whole_by_layer.mean()), 6
            ),
            "whole_request_subset_mean_std": round(
                float(whole.mean(axis=1).std()), 6
            ),
            "whole_request_layer_min": round(float(whole_by_layer.min()), 6),
            "whole_request_layer_max": round(float(whole_by_layer.max()), 6),
            "same_position_mean_unique_per_layer": round(
                float(same_position_by_layer.mean()), 6
            ),
            "same_position_subset_mean_std": round(
                float(same_position.mean(axis=1).std()), 6
            ),
            "same_position_layer_min": round(
                float(same_position_by_layer.min()), 6
            ),
            "same_position_layer_max": round(
                float(same_position_by_layer.max()), 6
            ),
        }
        summary_rows.append(row)
        for layer_id in range(layers):
            layer_rows.append(
                {
                    "k": k,
                    "layer_id": layer_id,
                    "whole_request_mean_unique": round(
                        float(whole_by_layer[layer_id]), 6
                    ),
                    "same_position_mean_unique": round(
                        float(same_position_by_layer[layer_id]), 6
                    ),
                }
            )

    summary: dict[str, object] = {
        "format": FORMAT,
        "source": {
            "file": str(dataset_path.resolve()),
            "size": dataset_path.stat().st_size,
            "sha256": sha256(dataset_path),
            "format": source_format,
            "manifest_sha256": manifest_sha256,
            "shape": list(expert_ids.shape),
        },
        "parameters": {
            "seed": seed,
            "monte_carlo_trials": trials,
            "k_values": k_values,
            "sampling": "requests sampled without replacement",
        },
        "axes": {
            "request_count": request_count,
            "output_positions": positions,
            "layer_count": layers,
            "top_k": top_k,
            "global_expert_count": 256,
        },
        "semantics": {
            "whole_request": (
                "Union IDs over all output positions in the selected requests "
                "for each layer; average over subsets and layers."
            ),
            "same_position": (
                "At each output position and layer, union IDs across selected "
                "requests; average over positions, subsets and layers."
            ),
        },
        "numpy_version": np.__version__,
        "results": summary_rows,
    }
    atomic_write(output_dir / "unique_experts_by_k.csv", csv_bytes(summary_rows))
    atomic_write(
        output_dir / "unique_experts_by_k_and_layer.csv",
        csv_bytes(layer_rows),
    )
    atomic_write(
        output_dir / "summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = analyze(
        args.dataset, args.output_dir, args.k_values, args.trials, args.seed
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
