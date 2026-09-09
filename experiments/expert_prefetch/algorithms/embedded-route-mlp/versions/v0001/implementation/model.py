"""EmbeddedRouteMLP v0001 model and objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NUM_EXPERTS, NUM_LAYERS, TOP_K, RouteBatch


@dataclass(frozen=True)
class EmbeddedRouteMLPConfig:
    history_tokens: int = 8
    current_previous_layers: int = 4
    lead_layers: int = 0
    route_embedding_dim: int = 32
    hidden_dim: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if (
            min(
                self.history_tokens,
                self.current_previous_layers,
                self.lead_layers,
            )
            < 0
        ):
            raise ValueError("causal context sizes must be non-negative")
        if self.route_embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def input_dim(self) -> int:
        route_slots = (
            NUM_LAYERS * self.history_tokens + self.current_previous_layers + 1
        )
        masks = self.history_tokens + self.current_previous_layers
        return route_slots * self.route_embedding_dim + masks

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["input_dim"] = self.input_dim
        return result


class EmbeddedRouteMLP(nn.Module):
    def __init__(self, config: EmbeddedRouteMLPConfig) -> None:
        super().__init__()
        self.config = config
        d = config.route_embedding_dim
        self.route_embedding = nn.Embedding(NUM_LAYERS * NUM_EXPERTS, d)
        self.target_layer_embedding = nn.Embedding(NUM_LAYERS, d)
        self.mlp = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, NUM_EXPERTS),
        )

    def _encode_routes(
        self, expert_ids: torch.Tensor, layer_ids: torch.Tensor
    ) -> torch.Tensor:
        logical_ids = layer_ids.unsqueeze(-1) * NUM_EXPERTS + expert_ids
        return self.route_embedding(logical_ids).mean(dim=-2)

    def forward(self, batch: RouteBatch) -> torch.Tensor:
        batch_size = batch.target_layer_ids.shape[0]
        device = batch.target_layer_ids.device
        dtype = self.route_embedding.weight.dtype

        if self.config.history_tokens:
            history_layers = torch.arange(NUM_LAYERS, device=device).view(
                1, 1, NUM_LAYERS
            )
            history = self._encode_routes(batch.history_expert_ids, history_layers)
            history = history * batch.history_valid_mask[:, :, None, None]
            history = history.reshape(batch_size, -1)
        else:
            history = torch.empty((batch_size, 0), dtype=dtype, device=device)

        if self.config.current_previous_layers:
            current = self._encode_routes(
                batch.current_layer_expert_ids, batch.current_layer_ids
            )
            current = current * batch.current_layer_valid_mask[:, :, None]
            current = current.reshape(batch_size, -1)
        else:
            current = torch.empty((batch_size, 0), dtype=dtype, device=device)

        target_layer = self.target_layer_embedding(batch.target_layer_ids)
        masks = torch.cat(
            (batch.history_valid_mask, batch.current_layer_valid_mask), dim=1
        ).to(dtype=dtype)
        features = torch.cat((history, current, target_layer, masks), dim=1)
        if features.shape[1] != self.config.input_dim:
            raise RuntimeError(
                f"feature width {features.shape[1]} != {self.config.input_dim}"
            )
        return self.mlp(features)


def set_cross_entropy(
    logits: torch.Tensor, target_expert_ids: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != NUM_EXPERTS:
        raise ValueError(f"unexpected logits shape: {tuple(logits.shape)}")
    if target_expert_ids.shape != (logits.shape[0], TOP_K):
        raise ValueError(f"unexpected target shape: {tuple(target_expert_ids.shape)}")
    # Keep the 256-way normalization in FP32 when the MLP runs under AMP.
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    selected = log_probabilities.gather(1, target_expert_ids)
    return -selected.mean()
