from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class MoeTraceSite:
    """Stable description of one routed-MoE decision point."""

    site_id: int
    module_path: str
    layer_id: int
    router_type: str
    hidden_size: int
    num_experts: int
    top_k: int


@dataclass(frozen=True)
class MoeTraceSiteLayout:
    """Flat-buffer offsets assigned to a :class:`MoeTraceSite`."""

    site: MoeTraceSite
    activation_q_offset: int
    activation_q_width: int
    activation_scale_offset: int
    activation_scale_width: int
    route_offset: int


@dataclass
class MoeTraceBatchOutput:
    """Device payload produced by one decode forward.

    ``map_device_tensors`` deliberately mirrors other scheduler result holders:
    the scheduler supplies the one lifetime-safe asynchronous D2H primitive.
    Host metadata never enters CUDA graphs.
    """

    request_ids: list[str]
    input_token_ids: torch.Tensor
    positions: torch.Tensor
    site_valid: torch.Tensor
    activation_q: Optional[torch.Tensor] = None
    activation_scales: Optional[torch.Tensor] = None
    expert_ids: Optional[torch.Tensor] = None
    expert_weights: Optional[torch.Tensor] = None
    release_callback: Optional[Callable[[], None]] = None

    def map_device_tensors(self, fn) -> None:
        self.input_token_ids = fn(self.input_token_ids)
        self.positions = fn(self.positions)
        self.site_valid = fn(self.site_valid)
        if self.activation_q is not None:
            self.activation_q = fn(self.activation_q)
        if self.activation_scales is not None:
            self.activation_scales = fn(self.activation_scales)
        if self.expert_ids is not None:
            self.expert_ids = fn(self.expert_ids)
        if self.expert_weights is not None:
            self.expert_weights = fn(self.expert_weights)

    def release(self) -> None:
        if self.release_callback is not None:
            callback, self.release_callback = self.release_callback, None
            callback()
