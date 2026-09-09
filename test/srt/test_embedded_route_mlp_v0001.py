from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_DIR = (
    REPO_ROOT
    / "experiments"
    / "expert_prefetch"
    / "algorithms"
    / "embedded-route-mlp"
    / "versions"
    / "v0001"
)
sys.path.insert(0, str(VERSION_DIR))

from implementation import (  # noqa: E402
    CausalRouteBatcher,
    EmbeddedRouteMLP,
    EmbeddedRouteMLPConfig,
    load_expert_ids,
    set_cross_entropy,
)


def make_routes(requests: int = 1) -> np.ndarray:
    request = np.arange(requests, dtype=np.int64)[:, None, None, None]
    position = np.arange(256, dtype=np.int64)[None, :, None, None]
    layer = np.arange(40, dtype=np.int64)[None, None, :, None]
    rank = np.arange(8, dtype=np.int64)[None, None, None, :]
    return ((request * 17 + position * 11 + layer * 7 + rank) % 256).astype(np.uint8)


def flat_index(request: int, position: int, layer: int) -> int:
    return (request * 255 + position - 1) * 40 + layer


def test_load_expert_ids_validates_split(tmp_path: Path) -> None:
    path = tmp_path / "split.npz"
    routes = make_routes(2)
    np.savez(
        path,
        expert_ids=routes,
        request_indices=np.asarray([3, 9], dtype=np.int32),
        prompt_sha256=np.asarray(["a" * 64, "b" * 64]),
    )
    loaded = load_expert_ids(path)
    assert np.array_equal(loaded, routes)


def test_causal_batch_has_expected_history_and_layers() -> None:
    routes = make_routes()
    batcher = CausalRouteBatcher(
        routes,
        history_tokens=2,
        current_previous_layers=2,
        lead_layers=0,
        device="cpu",
    )
    batch = batcher.make_batch(torch.tensor([flat_index(0, 3, 5)]))
    assert torch.equal(
        batch.history_expert_ids[0, 0], torch.from_numpy(routes[0, 2]).long()
    )
    assert torch.equal(
        batch.history_expert_ids[0, 1], torch.from_numpy(routes[0, 1]).long()
    )
    assert batch.history_valid_mask.tolist() == [[True, True]]
    assert batch.current_layer_ids.tolist() == [[4, 3]]
    assert torch.equal(
        batch.current_layer_expert_ids[0, 0], torch.from_numpy(routes[0, 3, 4]).long()
    )
    assert torch.equal(
        batch.target_expert_ids[0], torch.from_numpy(routes[0, 3, 5]).long()
    )


def test_missing_causal_context_is_masked() -> None:
    batcher = CausalRouteBatcher(
        make_routes(),
        history_tokens=3,
        current_previous_layers=2,
        lead_layers=0,
        device="cpu",
    )
    batch = batcher.make_batch(torch.tensor([flat_index(0, 1, 0)]))
    assert batch.history_valid_mask.tolist() == [[True, False, False]]
    assert batch.current_layer_valid_mask.tolist() == [[False, False]]


def test_target_and_future_routes_do_not_change_features() -> None:
    routes = make_routes()
    changed = routes.copy()
    changed[0, 3, 5] = ((changed[0, 3, 5].astype(np.int16) + 53) % 256).astype(np.uint8)
    changed[0, 3, 6:] = ((changed[0, 3, 6:].astype(np.int16) + 71) % 256).astype(
        np.uint8
    )
    changed[0, 4:] = ((changed[0, 4:].astype(np.int16) + 89) % 256).astype(np.uint8)
    kwargs = dict(
        history_tokens=2,
        current_previous_layers=3,
        lead_layers=0,
        device="cpu",
    )
    index = torch.tensor([flat_index(0, 3, 5)])
    original_batch = CausalRouteBatcher(routes, **kwargs).make_batch(index)
    changed_batch = CausalRouteBatcher(changed, **kwargs).make_batch(index)
    for name in (
        "history_expert_ids",
        "history_valid_mask",
        "current_layer_expert_ids",
        "current_layer_ids",
        "current_layer_valid_mask",
        "target_layer_ids",
    ):
        assert torch.equal(getattr(original_batch, name), getattr(changed_batch, name))
    assert not torch.equal(
        original_batch.target_expert_ids, changed_batch.target_expert_ids
    )


def test_lead_layers_removes_recent_current_layers() -> None:
    routes = make_routes()
    batcher = CausalRouteBatcher(
        routes,
        history_tokens=1,
        current_previous_layers=2,
        lead_layers=2,
        device="cpu",
    )
    batch = batcher.make_batch(torch.tensor([flat_index(0, 8, 7)]))
    assert batch.current_layer_ids.tolist() == [[4, 3]]
    assert torch.equal(
        batch.current_layer_expert_ids[0, 0], torch.from_numpy(routes[0, 8, 4]).long()
    )


@pytest.mark.parametrize("history_tokens,current_layers", [(0, 0), (2, 3)])
def test_model_shape_and_set_loss(history_tokens: int, current_layers: int) -> None:
    routes = make_routes()
    batcher = CausalRouteBatcher(
        routes,
        history_tokens=history_tokens,
        current_previous_layers=current_layers,
        lead_layers=0,
        device="cpu",
    )
    batch = batcher.make_batch(
        torch.tensor([flat_index(0, 2, 3), flat_index(0, 9, 17)])
    )
    config = EmbeddedRouteMLPConfig(
        history_tokens=history_tokens,
        current_previous_layers=current_layers,
        route_embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
    )
    model = EmbeddedRouteMLP(config)
    logits = model(batch)
    assert logits.shape == (2, 256)
    loss = set_cross_entropy(logits, batch.target_expert_ids)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.mlp[0].weight.grad is not None


def test_flat_index_bounds_are_checked() -> None:
    batcher = CausalRouteBatcher(
        make_routes(),
        history_tokens=1,
        current_previous_layers=1,
        lead_layers=0,
        device="cpu",
    )
    with pytest.raises(IndexError):
        batcher.make_batch(torch.tensor([batcher.num_samples]))
