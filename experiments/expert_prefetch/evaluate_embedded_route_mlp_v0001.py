#!/usr/bin/env python3
"""Evaluate a frozen EmbeddedRouteMLP v0001 checkpoint at candidate budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

NUM_LAYERS = 40
NUM_EXPERTS = 256
TRUE_TOP_K = 8
FIRST_TARGET_ROW = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with partial.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with partial.open("xb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        routes = archive["expert_ids"].copy()
        request_indices = archive["request_indices"].copy()
        trace_ids = archive["trace_ids"].copy()
    if routes.ndim != 4 or routes.shape[1:] != (256, 40, 8):
        raise ValueError(f"unexpected routes shape in {path}: {routes.shape}")
    if routes.dtype != np.uint8:
        raise ValueError(f"unexpected routes dtype in {path}: {routes.dtype}")
    return routes, request_indices, trace_ids


def per_layer_frequency(train_routes: np.ndarray, count: int = 32) -> np.ndarray:
    counts = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    for layer in range(NUM_LAYERS):
        values = train_routes[:, FIRST_TARGET_ROW:, layer, :].reshape(-1)
        counts[layer] = np.bincount(values, minlength=NUM_EXPERTS)
    expert_ids = np.arange(NUM_EXPERTS)
    return np.stack(
        [np.lexsort((expert_ids, -counts[layer]))[:count] for layer in range(40)]
    ).astype(np.uint8)


def fill_with_frequency(
    primary: np.ndarray, frequencies: np.ndarray, candidate_count: int
) -> np.ndarray:
    """Keep primary candidates in order, then append unique frequent experts."""

    if primary.shape[-2:] != (NUM_LAYERS, TRUE_TOP_K):
        raise ValueError(f"unexpected primary shape: {primary.shape}")
    if frequencies.shape[0] != NUM_LAYERS:
        raise ValueError(f"unexpected frequencies shape: {frequencies.shape}")
    if not TRUE_TOP_K <= candidate_count <= NUM_EXPERTS:
        raise ValueError("candidate_count must be in [8, 256]")
    flat = primary.reshape(-1, NUM_LAYERS, TRUE_TOP_K)
    result = np.empty(flat.shape[:-1] + (candidate_count,), dtype=np.uint8)
    for row in range(flat.shape[0]):
        for layer in range(NUM_LAYERS):
            seen: set[int] = set()
            output: list[int] = []
            for expert in np.concatenate((flat[row, layer], frequencies[layer])):
                value = int(expert)
                if value not in seen:
                    seen.add(value)
                    output.append(value)
                    if len(output) == candidate_count:
                        break
            if len(output) != candidate_count:
                raise ValueError("frequency candidates did not cover requested budget")
            result[row, layer] = output
    return result.reshape(primary.shape[:-1] + (candidate_count,))


def summarize_candidates(
    candidates: np.ndarray, targets: np.ndarray, *, history_tokens: int
) -> dict[str, Any]:
    if candidates.shape[:-1] != targets.shape[:-1]:
        raise ValueError("candidate/target prefix shape mismatch")
    if targets.shape[-2:] != (NUM_LAYERS, TRUE_TOP_K):
        raise ValueError(f"unexpected target shape: {targets.shape}")
    sorted_candidates = np.sort(candidates, axis=-1)
    if np.any(np.diff(sorted_candidates.astype(np.int16), axis=-1) == 0):
        raise ValueError("candidate rows contain duplicate experts")
    matches = (candidates[..., :, None] == targets[..., None, :]).any(axis=-1)
    hits = matches.sum(axis=-1)
    candidate_count = candidates.shape[-1]

    def summarize_hits(values: np.ndarray) -> dict[str, Any]:
        samples = values.size
        useful = int(values.sum())
        histogram = np.bincount(values.reshape(-1), minlength=TRUE_TOP_K + 1)
        return {
            "samples": samples,
            "candidate_count": candidate_count,
            "actual_experts_per_sample": TRUE_TOP_K,
            "useful_candidates": useful,
            "incorrect_candidates": samples * candidate_count - useful,
            "missed_actual_experts": samples * TRUE_TOP_K - useful,
            "mean_useful_candidates": useful / samples,
            "mean_incorrect_candidates": candidate_count - useful / samples,
            "mean_missed_actual_experts": TRUE_TOP_K - useful / samples,
            "recall": useful / (samples * TRUE_TOP_K),
            "precision": useful / (samples * candidate_count),
            "fully_covered_rate": float(np.mean(values == TRUE_TOP_K)),
            "hit_count_histogram": {
                str(index): int(count) for index, count in enumerate(histogram)
            },
        }

    result = summarize_hits(hits)
    result["offload_layers_1_38"] = summarize_hits(hits[..., 1:39])
    result["per_layer"] = {
        str(layer): summarize_hits(hits[..., layer]) for layer in range(NUM_LAYERS)
    }
    # rows 1..history_tokens-1 have fewer than t visible trace rows. Row t is
    # the first position with a complete t-row history including row 0.
    positions = np.arange(FIRST_TARGET_ROW, 256)
    cold_mask = positions < history_tokens
    result["cold_start"] = (
        summarize_hits(hits[:, cold_mask]) if np.any(cold_mask) else None
    )
    result["steady_state"] = summarize_hits(hits[:, ~cold_mask])
    return result


def summarize_candidate_prefixes(
    candidates: np.ndarray,
    targets: np.ndarray,
    *,
    candidate_counts: list[int],
    history_tokens: int,
) -> dict[str, dict[str, Any]]:
    return {
        str(count): summarize_candidates(
            candidates[..., :count], targets, history_tokens=history_tokens
        )
        for count in candidate_counts
    }


def predict_candidates(
    model: torch.nn.Module,
    batcher: Any,
    *,
    candidate_count: int,
    batch_size: int,
    amp: bool,
) -> tuple[np.ndarray, float]:
    model.eval()
    output = np.empty((batcher.num_samples, candidate_count), dtype=np.uint8)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, batcher.num_samples, batch_size):
            stop = min(start + batch_size, batcher.num_samples)
            indices = torch.arange(start, stop, device=batcher.device)
            batch = batcher.make_batch(indices)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp
                else nullcontext()
            )
            with context:
                logits = model(batch)
            output[start:stop] = (
                torch.topk(logits, candidate_count, dim=-1)
                .indices.to(dtype=torch.uint8)
                .cpu()
                .numpy()
            )
    elapsed = time.perf_counter() - started
    return output.reshape(batcher.num_requests, 255, 40, candidate_count), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--validation-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument(
        "--candidate-counts", type=int, nargs="+", default=[8, 16, 24, 32]
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-compute-capability", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    args = parser.parse_args()
    candidate_counts = sorted(set(args.candidate_counts))
    if not candidate_counts or any(
        count < TRUE_TOP_K or count > NUM_EXPERTS for count in candidate_counts
    ):
        parser.error("candidate counts must be in [8, 256]")
    max_candidate_count = max(candidate_counts)

    version_dir = args.version_dir.resolve()
    sys.path.insert(0, str(version_dir))
    from implementation import (  # pylint: disable=import-outside-toplevel
        CausalRouteBatcher,
        EmbeddedRouteMLP,
        EmbeddedRouteMLPConfig,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA GPU must be visible")
        capability = torch.cuda.get_device_capability(0)
        capability_text = f"{capability[0]}.{capability[1]}"
        if (
            args.require_compute_capability
            and capability_text != args.require_compute_capability
        ):
            raise RuntimeError(
                f"compute capability {capability_text} != required "
                f"{args.require_compute_capability}"
            )
    else:
        capability_text = None

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("algorithm_version") != "v0001":
        raise ValueError("checkpoint is not EmbeddedRouteMLP v0001")
    model_values = dict(checkpoint["model_config"])
    model_values.pop("input_dim", None)
    config = EmbeddedRouteMLPConfig(**model_values)
    model = EmbeddedRouteMLP(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    train_routes, _, _ = load_split(args.train_npz)
    validation_routes, validation_request_indices, validation_trace_ids = load_split(
        args.validation_npz
    )
    test_routes, test_request_indices, test_trace_ids = load_split(args.test_npz)
    frequencies = per_layer_frequency(train_routes, count=32)
    split_values = {
        "validation": (
            validation_routes,
            validation_request_indices,
            validation_trace_ids,
        ),
        "test": (test_routes, test_request_indices, test_trace_ids),
    }
    metrics: dict[str, Any] = {}
    prediction_arrays: dict[str, np.ndarray] = {
        "format": np.asarray("SGLANG-ROUTE-PREDICTION-CANDIDATES-v1"),
        "candidate_counts": np.asarray(candidate_counts, dtype=np.int32),
        "max_candidate_count": np.asarray(max_candidate_count, dtype=np.int32),
        "checkpoint_sha256": np.asarray(sha256(checkpoint_path)),
    }
    amp = device.type == "cuda"
    for split_name, (routes, request_indices, trace_ids) in split_values.items():
        batcher = CausalRouteBatcher(
            routes,
            history_tokens=config.history_tokens,
            current_previous_layers=config.current_previous_layers,
            lead_layers=config.lead_layers,
            device=device,
        )
        predictions, elapsed = predict_candidates(
            model,
            batcher,
            candidate_count=max_candidate_count,
            batch_size=args.batch_size,
            amp=amp,
        )
        targets = routes[:, FIRST_TARGET_ROW:, :, :]
        last_token = fill_with_frequency(
            routes[:, :-1, :, :], frequencies, max_candidate_count
        )
        previous_layer_primary = np.empty_like(targets)
        previous_layer_primary[:, :, 0, :] = frequencies[0, :TRUE_TOP_K]
        previous_layer_primary[:, :, 1:, :] = routes[:, 1:, :-1, :]
        previous_layer = fill_with_frequency(
            previous_layer_primary, frequencies, max_candidate_count
        )
        frequency_prediction = np.broadcast_to(
            frequencies[:, :max_candidate_count], predictions.shape
        )
        metrics[split_name] = {
            "predictor": summarize_candidate_prefixes(
                predictions,
                targets,
                candidate_counts=candidate_counts,
                history_tokens=config.history_tokens,
            ),
            "baselines": {
                "per_layer_frequency": summarize_candidate_prefixes(
                    frequency_prediction,
                    targets,
                    candidate_counts=candidate_counts,
                    history_tokens=config.history_tokens,
                ),
                "same_layer_previous_token_plus_frequency": summarize_candidate_prefixes(
                    last_token,
                    targets,
                    candidate_counts=candidate_counts,
                    history_tokens=config.history_tokens,
                ),
                "current_token_previous_layer_plus_frequency": summarize_candidate_prefixes(
                    previous_layer,
                    targets,
                    candidate_counts=candidate_counts,
                    history_tokens=config.history_tokens,
                ),
            },
            "inference_seconds": elapsed,
            "batched_samples_per_second": batcher.num_samples / elapsed,
        }
        prediction_arrays[f"{split_name}_candidates"] = predictions
        prediction_arrays[f"{split_name}_request_indices"] = request_indices
        prediction_arrays[f"{split_name}_trace_ids"] = trace_ids

    atomic_npz(args.predictions_npz, prediction_arrays)
    result = {
        "format": "SGLANG-EMBEDDED-ROUTE-MLP-CANDIDATE-EVAL-v1",
        "algorithm": "embedded-route-mlp",
        "algorithm_version": "v0001",
        "candidate_counts": candidate_counts,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "source_epoch": checkpoint["epoch"],
        },
        "model_config": config.to_dict(),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "compute_capability": capability_text,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "amp": amp,
        },
        "data": {
            "train": {
                "path": str(args.train_npz.resolve()),
                "sha256": sha256(args.train_npz),
            },
            "validation": {
                "path": str(args.validation_npz.resolve()),
                "sha256": sha256(args.validation_npz),
            },
            "test": {
                "path": str(args.test_npz.resolve()),
                "sha256": sha256(args.test_npz),
            },
        },
        "metrics": metrics,
    }
    atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
