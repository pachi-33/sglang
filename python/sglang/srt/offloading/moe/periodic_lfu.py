"""Periodic, layer-local LFU with rank-wide admission limits.

Online decisions stay on the tensor device. Python loops only over static layer
ranges; no routing/popularity tensor is read by the host. Accelerator bindings
must validate capture support and budget the sorting temporaries/workspace.
"""

import json
import math
from dataclasses import dataclass
from itertools import islice, zip_longest
from typing import Optional

from .config import ExpertCatalog
from .interfaces import CachePolicy


@dataclass(frozen=True)
class PeriodicLFUConfig:
    offline_stats_path: Optional[str] = None
    update_interval: int = 4
    min_usage: float = 2.0
    replacement_min_gain: float = 1.5
    max_swaps: int = 96
    usage_decay: float = 0.7

    def __post_init__(self):
        if self.offline_stats_path is not None and (
            not isinstance(self.offline_stats_path, str) or not self.offline_stats_path
        ):
            raise ValueError("offline_stats_path must be a nonempty path")
        if type(self.update_interval) is not int or self.update_interval <= 0:
            raise ValueError("update_interval must be a positive integer")
        if type(self.max_swaps) is not int or self.max_swaps < 0:
            raise ValueError("max_swaps must be a nonnegative integer")
        for name in ("min_usage", "replacement_min_gain"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.usage_decay) or not 0 <= self.usage_decay <= 1:
            raise ValueError("usage_decay must be finite and in [0, 1]")


@dataclass
class PeriodicLFUState:
    usage: object
    step_mod: object
    update_due: object


class PeriodicLFUPolicy(CachePolicy):
    def __init__(self, config: PeriodicLFUConfig, catalog: ExpertCatalog):
        self.config = config
        self.catalog = catalog
        offset = 0
        ranges = []
        for layer in catalog.offloaded_layers:
            ranges.append((offset, offset + layer.local_experts))
            offset += layer.local_experts
        # Catalog order is layer_id order, also the global pair tie-break order.
        self.layer_ranges = tuple(ranges)

    def initial_keys(self, capacity: int, model_id=None) -> tuple[int, ...]:
        """Share slots across layers, optionally ranking each by offline counts.

        Frequencies use global expert IDs. Resident layers and foreign EP experts
        are excluded by the catalog. Online usage is not seeded with this data.
        The backend loads/publishes complete bundles before serving.
        """
        if type(capacity) is not int or not 0 <= capacity <= self.catalog.num_experts:
            raise ValueError("Hot capacity must be within the eligible expert count")
        if capacity == 0:
            return ()
        layers = {}
        if self.config.offline_stats_path is not None:
            with open(self.config.offline_stats_path, encoding="utf-8") as stream:
                profile = json.load(stream)
            if not isinstance(profile, dict) or not isinstance(
                profile.get("layers"), dict
            ):
                raise ValueError("Offline profile must contain a layers object")
            if model_id is not None and profile.get("model_id") != model_id:
                raise ValueError(
                    "Offline profile model_id does not match offline_model_id"
                )
            layers = profile["layers"]
        ranked = []
        offset = 0
        for layer in self.catalog.offloaded_layers:
            frequencies = layers.get(str(layer.layer_id), {})
            if not isinstance(frequencies, dict):
                raise ValueError("Layer frequencies must be an object")
            counts = []
            for local_id in range(layer.local_experts):
                frequency = frequencies.get(str(layer.global_start + local_id), 0)
                if (
                    type(frequency) not in (int, float)
                    or not math.isfinite(frequency)
                    or frequency < 0
                ):
                    raise ValueError(
                        "Offline frequencies must be finite and nonnegative"
                    )
                counts.append(frequency)
            ranked.append(
                sorted(
                    range(offset, offset + layer.local_experts),
                    key=lambda key: (-counts[key - offset], key),
                )
            )
            offset += layer.local_experts
        # One slot per layer each round. Full layers drop out; a small capacity
        # favors lower layer IDs. Offline counts change experts, not layer quotas.
        keys = (key for row in zip_longest(*ranked) for key in row if key is not None)
        return tuple(sorted(islice(keys, capacity)))

    def allocate_state(self, num_experts, device):
        import torch

        if num_experts != self.catalog.num_experts:
            raise ValueError("State size must match the expert catalog")
        return PeriodicLFUState(
            usage=torch.zeros(num_experts, dtype=torch.float32, device=device),
            step_mod=torch.zeros((), dtype=torch.int64, device=device),
            update_due=torch.zeros((), dtype=torch.bool, device=device),
        )

    def update_statistics(self, state, step_counts, update_enabled):
        import torch

        next_step = (state.step_mod + 1).remainder(self.config.update_interval)
        state.usage.add_(torch.where(update_enabled, step_counts, 0))
        state.step_mod.copy_(torch.where(update_enabled, next_step, state.step_mod))
        state.update_due.copy_(update_enabled & (next_step == 0))

    def select_hot_set(self, state, key_to_slot, target_keys):
        import torch

        count = self.catalog.num_experts
        capacity = target_keys.numel()
        if capacity > count:
            raise ValueError("Hot capacity exceeds the eligible expert count")
        if state.usage.numel() != count or key_to_slot.numel() != count:
            raise ValueError("Policy tensors must match the expert catalog")
        # These branches depend only on static capacity, never on device values.
        if capacity == 0:
            return
        resident = key_to_slot >= 0
        candidates, victims = [], []
        for start, end in self.layer_ranges:
            usage = state.usage[start:end]
            hot = resident[start:end]
            # Fixed-size permutations: invalid entries sort to the end. Their
            # pairs are masked out instead of constructing variable-size arrays.
            cold_order = torch.argsort(
                torch.where(~hot, usage, -torch.inf), descending=True, stable=True
            )
            hot_order = torch.argsort(torch.where(hot, usage, torch.inf), stable=True)
            candidates.append(cold_order + start)
            victims.append(hot_order + start)

        candidate = torch.cat(candidates)
        victim = torch.cat(victims)
        gain = state.usage[candidate] - state.usage[victim]
        valid = (
            ~resident[candidate]
            & resident[victim]
            & (state.usage[candidate] >= self.config.min_usage)
            & (gain >= self.config.replacement_min_gain)
        )
        # Stable sorting preserves layer_id, candidate key, victim key on ties:
        # within each layer equal gains imply equal candidate/victim usages.
        order = torch.argsort(
            torch.where(valid, gain, -torch.inf), descending=True, stable=True
        )
        chosen = order[: self.config.max_swaps]
        swaps = (valid[chosen] & state.update_due).to(torch.int64)
        # Each selected pair replaces a distinct victim in its existing slot.
        # This avoids rebuilding and sorting an E-entry membership vector.
        # The public result remains a sorted set, not a slot-placement request.
        keys = torch.arange(count, device=state.usage.device, dtype=torch.int64)
        slots = key_to_slot.clamp_min(0)
        placed = torch.zeros(capacity, dtype=torch.int64, device=keys.device)
        placed.scatter_add_(0, slots, torch.where(resident, keys + 1, 0))
        placed.scatter_add_(
            0,
            key_to_slot[victim[chosen]].clamp_min(0),
            swaps * (candidate[chosen] - victim[chosen]),
        )
        selected = torch.sort(torch.where(placed > 0, placed - 1, count)).values
        target_keys.copy_(torch.where(selected < count, selected, -1))

    def finish_step(self, state):
        import torch

        # Separate from pure selection. Call once after select_hot_set, even on
        # empty-update rounds. Clearing due makes repeated finish calls harmless.
        state.usage.copy_(
            torch.where(
                state.update_due, state.usage * self.config.usage_decay, state.usage
            )
        )
        state.update_due.zero_()

    def copy_state(self, source, destination):
        destination.usage.copy_(source.usage)
        destination.step_mod.copy_(source.step_mod)
        destination.update_due.copy_(source.update_due)
