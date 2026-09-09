"""Causal samples for EmbeddedRouteMLP v0001."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

NUM_LAYERS = 40
NUM_EXPERTS = 256
TOP_K = 8
TRACE_ROWS = 256
FIRST_TARGET_ROW = 1
TARGET_ROWS = TRACE_ROWS - FIRST_TARGET_ROW


def load_expert_ids(path: Path) -> np.ndarray:
    """Load and validate one request-level EXP-0001 split."""

    with np.load(path, allow_pickle=False) as archive:
        if "expert_ids" not in archive.files:
            raise ValueError(f"expert_ids is missing from {path}")
        expert_ids = archive["expert_ids"].copy()
        request_indices = archive["request_indices"].copy()
        prompt_sha256 = archive["prompt_sha256"].copy()
    if expert_ids.ndim != 4 or expert_ids.shape[1:] != (
        TRACE_ROWS,
        NUM_LAYERS,
        TOP_K,
    ):
        raise ValueError(f"unexpected expert_ids shape: {expert_ids.shape}")
    if expert_ids.dtype != np.uint8:
        raise ValueError(f"unexpected expert_ids dtype: {expert_ids.dtype}")
    if request_indices.shape != (expert_ids.shape[0],):
        raise ValueError("request_indices length does not match expert_ids")
    if prompt_sha256.shape != (expert_ids.shape[0],):
        raise ValueError("prompt_sha256 length does not match expert_ids")
    if len(set(request_indices.tolist())) != expert_ids.shape[0]:
        raise ValueError("request_indices are not unique")
    if len(set(prompt_sha256.tolist())) != expert_ids.shape[0]:
        raise ValueError("prompt SHA-256 values are not unique")
    return expert_ids


@dataclass(frozen=True)
class RouteBatch:
    history_expert_ids: torch.Tensor
    history_valid_mask: torch.Tensor
    current_layer_expert_ids: torch.Tensor
    current_layer_ids: torch.Tensor
    current_layer_valid_mask: torch.Tensor
    target_layer_ids: torch.Tensor
    target_expert_ids: torch.Tensor
    request_ids: torch.Tensor
    positions: torch.Tensor


class CausalRouteBatcher:
    """Vectorized causal feature extraction over a complete trace split."""

    def __init__(
        self,
        expert_ids: np.ndarray | torch.Tensor,
        *,
        history_tokens: int,
        current_previous_layers: int,
        lead_layers: int,
        device: torch.device | str,
    ) -> None:
        if history_tokens < 0 or current_previous_layers < 0 or lead_layers < 0:
            raise ValueError("causal context sizes must be non-negative")
        tensor = torch.as_tensor(expert_ids, dtype=torch.long, device=device)
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
            TRACE_ROWS,
            NUM_LAYERS,
            TOP_K,
        ):
            raise ValueError(f"unexpected expert_ids shape: {tuple(tensor.shape)}")
        if tensor.numel() and (tensor.min().item() < 0 or tensor.max().item() >= 256):
            raise ValueError("expert ID outside 0..255")
        self.expert_ids = tensor
        self.history_tokens = history_tokens
        self.current_previous_layers = current_previous_layers
        self.lead_layers = lead_layers
        self.device = tensor.device
        self.num_requests = tensor.shape[0]
        self.num_samples = self.num_requests * TARGET_ROWS * NUM_LAYERS

    def decode_indices(self, flat_indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if (
            flat_indices.device.type == "cpu"
            and flat_indices.numel()
            and (
                flat_indices.min().item() < 0
                or flat_indices.max().item() >= self.num_samples
            )
        ):
            raise IndexError("flat sample index out of range")
        flat_indices = flat_indices.to(device=self.device, dtype=torch.long)
        target_layers = flat_indices.remainder(NUM_LAYERS)
        token_indices = torch.div(flat_indices, NUM_LAYERS, rounding_mode="floor")
        positions = token_indices.remainder(TARGET_ROWS) + FIRST_TARGET_ROW
        request_ids = torch.div(token_indices, TARGET_ROWS, rounding_mode="floor")
        return request_ids, positions, target_layers

    def make_batch(self, flat_indices: torch.Tensor) -> RouteBatch:
        request_ids, positions, target_layers = self.decode_indices(flat_indices)
        batch_size = flat_indices.numel()

        if self.history_tokens:
            offsets = torch.arange(
                1,
                self.history_tokens + 1,
                dtype=torch.long,
                device=self.device,
            )
            history_positions = positions[:, None] - offsets[None, :]
            history_valid = history_positions >= 0
            history_positions = history_positions.clamp_min(0)
            all_layers = torch.arange(NUM_LAYERS, device=self.device)
            history_ids = self.expert_ids[
                request_ids[:, None, None],
                history_positions[:, :, None],
                all_layers[None, None, :],
            ]
        else:
            history_ids = torch.empty(
                (batch_size, 0, NUM_LAYERS, TOP_K),
                dtype=torch.long,
                device=self.device,
            )
            history_valid = torch.empty(
                (batch_size, 0), dtype=torch.bool, device=self.device
            )

        if self.current_previous_layers:
            offsets = torch.arange(
                1,
                self.current_previous_layers + 1,
                dtype=torch.long,
                device=self.device,
            )
            current_layers = (
                target_layers[:, None] - self.lead_layers - offsets[None, :]
            )
            current_valid = current_layers >= 0
            current_layers_clamped = current_layers.clamp_min(0)
            current_ids = self.expert_ids[
                request_ids[:, None],
                positions[:, None],
                current_layers_clamped,
            ]
        else:
            current_ids = torch.empty(
                (batch_size, 0, TOP_K), dtype=torch.long, device=self.device
            )
            current_layers_clamped = torch.empty(
                (batch_size, 0), dtype=torch.long, device=self.device
            )
            current_valid = torch.empty(
                (batch_size, 0), dtype=torch.bool, device=self.device
            )

        target_ids = self.expert_ids[request_ids, positions, target_layers]
        return RouteBatch(
            history_expert_ids=history_ids,
            history_valid_mask=history_valid,
            current_layer_expert_ids=current_ids,
            current_layer_ids=current_layers_clamped,
            current_layer_valid_mask=current_valid,
            target_layer_ids=target_layers,
            target_expert_ids=target_ids,
            request_ids=request_ids,
            positions=positions,
        )
