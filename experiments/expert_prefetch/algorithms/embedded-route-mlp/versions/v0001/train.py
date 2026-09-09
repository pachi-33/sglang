#!/usr/bin/env python3
"""Train and evaluate EmbeddedRouteMLP v0001 on EXP-0001 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from implementation import (
    CausalRouteBatcher,
    EmbeddedRouteMLP,
    EmbeddedRouteMLPConfig,
    load_expert_ids,
    set_cross_entropy,
)
from implementation.data import NUM_EXPERTS, NUM_LAYERS, TOP_K

FORMAT = "SGLANG-EMBEDDED-ROUTE-MLP-EXPERIMENT-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if partial.exists():
        raise FileExistsError(partial)
    with partial.open("xb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def atomic_checkpoint(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with partial.open("xb") as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics_from_top_candidates(
    candidates: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    if candidates.shape[:-1] != targets.shape[:-1] or targets.shape[-1] != TOP_K:
        raise ValueError("candidate/target shape mismatch")
    result: dict[str, Any] = {"candidate_count": int(candidates.shape[-1])}
    for count in (8, 16, 32):
        if candidates.shape[-1] < count:
            continue
        available = min(count, candidates.shape[-1])
        matches = (candidates[..., :available, None] == targets[..., None, :]).any(
            axis=-1
        )
        hits = matches.sum(axis=-1)
        result[f"recall_at_{count}"] = float(hits.sum() / targets.size)
        if count == 8 and available == 8:
            result["exact_set_accuracy"] = float(np.mean(hits == TOP_K))
            result["mean_jaccard"] = float(np.mean(hits / (2 * TOP_K - hits)))
            result["offload_recall_at_8"] = float(
                hits[..., 1:39].sum() / targets[..., 1:39, :].size
            )
            result["per_layer"] = {
                str(layer): {
                    "recall_at_8": float(
                        hits[..., layer].sum() / targets[..., layer, :].size
                    )
                }
                for layer in range(NUM_LAYERS)
            }
    return result


def frequency_candidates(train_routes: np.ndarray, count: int = 32) -> np.ndarray:
    counts = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
    for layer in range(NUM_LAYERS):
        values = train_routes[:, 1:, layer, :].reshape(-1)
        counts[layer] = np.bincount(values, minlength=NUM_EXPERTS)
    expert_ids = np.arange(NUM_EXPERTS)
    return np.stack(
        [
            np.lexsort((expert_ids, -counts[layer]))[:count]
            for layer in range(NUM_LAYERS)
        ]
    )


def evaluate_baselines(
    train_routes: np.ndarray, split_routes: np.ndarray
) -> dict[str, dict[str, Any]]:
    targets = split_routes[:, 1:, :, :]
    frequencies = frequency_candidates(train_routes)
    frequency_prediction = np.broadcast_to(
        frequencies, targets.shape[:-1] + (frequencies.shape[-1],)
    )

    last_token = split_routes[:, :-1, :, :]
    previous_layer = np.empty_like(targets)
    previous_layer[:, :, 0, :] = frequencies[0, :TOP_K]
    previous_layer[:, :, 1:, :] = split_routes[:, 1:, :-1, :]
    return {
        "per_layer_frequency": _metrics_from_top_candidates(
            frequency_prediction, targets
        ),
        "same_layer_previous_token": _metrics_from_top_candidates(last_token, targets),
        "current_token_previous_layer": _metrics_from_top_candidates(
            previous_layer, targets
        ),
    }


def _autocast(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def evaluate_model(
    model: EmbeddedRouteMLP,
    batcher: CausalRouteBatcher,
    *,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    hit_counts = {8: 0, 16: 0, 32: 0}
    exact_count = 0
    jaccard_sum = 0.0
    layer_hits = torch.zeros(NUM_LAYERS, dtype=torch.long, device=batcher.device)
    layer_samples = torch.zeros(NUM_LAYERS, dtype=torch.long, device=batcher.device)

    started = time.perf_counter()
    for start in range(0, batcher.num_samples, batch_size):
        stop = min(start + batch_size, batcher.num_samples)
        indices = torch.arange(start, stop, device=batcher.device)
        batch = batcher.make_batch(indices)
        with _autocast(batcher.device, amp):
            logits = model(batch)
        loss = set_cross_entropy(logits, batch.target_expert_ids)
        candidates = torch.topk(logits, k=32, dim=-1).indices
        for count in hit_counts:
            matches = (
                candidates[:, :count, None] == batch.target_expert_ids[:, None, :]
            ).any(dim=-1)
            hit_counts[count] += int(matches.sum().item())
            if count == TOP_K:
                intersections = matches.sum(dim=-1)
                exact_count += int((intersections == TOP_K).sum().item())
                jaccard_sum += float(
                    (intersections.float() / (2 * TOP_K - intersections)).sum().item()
                )
                layer_hits.scatter_add_(0, batch.target_layer_ids, intersections)
        layer_samples.scatter_add_(
            0,
            batch.target_layer_ids,
            torch.ones_like(batch.target_layer_ids),
        )
        current_samples = stop - start
        total_loss += float(loss.item()) * current_samples
        total_samples += current_samples

    elapsed = time.perf_counter() - started
    per_layer = {
        str(layer): {
            "recall_at_8": float(
                layer_hits[layer].item() / (TOP_K * layer_samples[layer].item())
            ),
            "samples": int(layer_samples[layer].item()),
        }
        for layer in range(NUM_LAYERS)
    }
    result: dict[str, Any] = {
        "loss": total_loss / total_samples,
        "exact_set_accuracy": exact_count / total_samples,
        "mean_jaccard": jaccard_sum / total_samples,
        "samples": total_samples,
        "seconds": elapsed,
        "samples_per_second": total_samples / elapsed,
        "per_layer": per_layer,
        "offload_recall_at_8": float(
            layer_hits[1:39].sum().item() / (TOP_K * layer_samples[1:39].sum().item())
        ),
    }
    for count, hits in hit_counts.items():
        result[f"recall_at_{count}"] = hits / (TOP_K * total_samples)
    return result


def train_epoch(
    model: EmbeddedRouteMLP,
    batcher: CausalRouteBatcher,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    *,
    batch_size: int,
    amp: bool,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(batcher.num_samples, device=batcher.device)
    total_loss = 0.0
    total_samples = 0
    started = time.perf_counter()
    for start in range(0, batcher.num_samples, batch_size):
        indices = permutation[start : start + batch_size]
        batch = batcher.make_batch(indices)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(batcher.device, amp):
            logits = model(batch)
            loss = set_cross_entropy(logits, batch.target_expert_ids)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        current_samples = indices.numel()
        total_loss += float(loss.item()) * current_samples
        total_samples += current_samples
    elapsed = time.perf_counter() - started
    return {
        "loss": total_loss / total_samples,
        "seconds": elapsed,
        "samples_per_second": total_samples / elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--validation-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--epoch-log-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history-tokens", type=int, default=8)
    parser.add_argument("--current-previous-layers", type=int, default=4)
    parser.add_argument("--lead-layers", type=int, default=0)
    parser.add_argument("--route-embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--require-compute-capability", default="")
    args = parser.parse_args()

    if min(args.batch_size, args.eval_batch_size, args.max_epochs) <= 0:
        parser.error("batch sizes and max epochs must be positive")
    if args.results_json.exists() or args.checkpoint.exists():
        parser.error("refusing to overwrite results or checkpoint")
    if args.epoch_log_jsonl.exists():
        parser.error("refusing to append to an existing epoch log")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("the experiment requires exactly one visible CUDA GPU")
        capability = torch.cuda.get_device_capability(0)
        capability_text = f"{capability[0]}.{capability[1]}"
        if (
            args.require_compute_capability
            and capability_text != args.require_compute_capability
        ):
            raise RuntimeError(
                f"compute capability {capability_text} != "
                f"required {args.require_compute_capability}"
            )
    else:
        capability_text = None

    train_path = args.train_npz.resolve()
    validation_path = args.validation_npz.resolve()
    test_path = args.test_npz.resolve()
    train_routes_np = load_expert_ids(train_path)
    validation_routes_np = load_expert_ids(validation_path)
    test_routes_np = load_expert_ids(test_path)

    model_config = EmbeddedRouteMLPConfig(
        history_tokens=args.history_tokens,
        current_previous_layers=args.current_previous_layers,
        lead_layers=args.lead_layers,
        route_embedding_dim=args.route_embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = EmbeddedRouteMLP(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_batcher = CausalRouteBatcher(
        train_routes_np,
        history_tokens=args.history_tokens,
        current_previous_layers=args.current_previous_layers,
        lead_layers=args.lead_layers,
        device=device,
    )
    validation_batcher = CausalRouteBatcher(
        validation_routes_np,
        history_tokens=args.history_tokens,
        current_previous_layers=args.current_previous_layers,
        lead_layers=args.lead_layers,
        device=device,
    )
    test_batcher = CausalRouteBatcher(
        test_routes_np,
        history_tokens=args.history_tokens,
        current_previous_layers=args.current_previous_layers,
        lead_layers=args.lead_layers,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp = device.type == "cuda" and not args.disable_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    device_info = {
        "type": device.type,
        "name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "compute_capability": capability_text,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    run_config: dict[str, Any] = {
        "format": FORMAT,
        "algorithm": "embedded-route-mlp",
        "algorithm_version": "v0001",
        "seed": args.seed,
        "model": model_config.to_dict(),
        "parameter_count": parameter_count,
        "training": {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "amp": amp,
        },
        "data": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "validation": {
                "path": str(validation_path),
                "sha256": sha256(validation_path),
            },
            "test": {"path": str(test_path), "sha256": sha256(test_path)},
            "target_rows": [1, 255],
            "target_layers": [0, 39],
        },
        "device": device_info,
    }
    print(json.dumps({"event": "run_start", **run_config}, sort_keys=True), flush=True)

    baselines = {
        "validation": evaluate_baselines(train_routes_np, validation_routes_np),
        "test": evaluate_baselines(train_routes_np, test_routes_np),
    }
    print(json.dumps({"event": "baselines", **baselines}, sort_keys=True), flush=True)

    best_recall = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_records: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_epoch(
            model,
            train_batcher,
            optimizer,
            scaler,
            batch_size=args.batch_size,
            amp=amp,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        validation_metrics = evaluate_model(
            model,
            validation_batcher,
            batch_size=args.eval_batch_size,
            amp=amp,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        append_jsonl(args.epoch_log_jsonl, record)
        epoch_records.append(record)
        print(json.dumps({"event": "epoch", **record}, sort_keys=True), flush=True)

        recall = validation_metrics["recall_at_8"]
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_checkpoint(
                args.checkpoint,
                {
                    "format": FORMAT,
                    "algorithm": "embedded-route-mlp",
                    "algorithm_version": "v0001",
                    "model_config": model_config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "run_config": run_config,
                },
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.early_stopping_patience:
            break

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    frozen_validation = evaluate_model(
        model, validation_batcher, batch_size=args.eval_batch_size, amp=amp
    )
    # This is the sole test evaluation after selecting the best validation epoch.
    test_metrics = evaluate_model(
        model, test_batcher, batch_size=args.eval_batch_size, amp=amp
    )
    elapsed = time.perf_counter() - run_started
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
    else:
        peak_memory = peak_reserved = None
    result = {
        **run_config,
        "status": "completed",
        "best_epoch": best_epoch,
        "epochs_completed": len(epoch_records),
        "duration_seconds": elapsed,
        "baselines": baselines,
        "validation": frozen_validation,
        "test": test_metrics,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "size": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
        },
        "epoch_log": {
            "path": str(args.epoch_log_jsonl.resolve()),
            "size": args.epoch_log_jsonl.stat().st_size,
            "sha256": sha256(args.epoch_log_jsonl),
        },
        "gpu_peak_memory_allocated": peak_memory,
        "gpu_peak_memory_reserved": peak_reserved,
    }
    atomic_json(args.results_json, result)
    print(json.dumps({"event": "run_complete", **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
