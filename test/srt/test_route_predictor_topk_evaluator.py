from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expert_prefetch"))

from evaluate_embedded_route_mlp_v0001 import (  # noqa: E402
    fill_with_frequency,
    summarize_candidate_prefixes,
    summarize_candidates,
)


def test_fill_with_frequency_preserves_primary_and_is_unique() -> None:
    primary = np.zeros((1, 1, 40, 8), dtype=np.uint8)
    primary[...] = np.asarray([5, 3, 7, 1, 9, 2, 4, 6], dtype=np.uint8)
    frequencies = np.broadcast_to(np.arange(32, dtype=np.uint8), (40, 32)).copy()
    result = fill_with_frequency(primary, frequencies, 16)
    assert result.shape == (1, 1, 40, 16)
    assert np.array_equal(result[0, 0, 17, :8], primary[0, 0, 17])
    assert len(set(result[0, 0, 17].tolist())) == 16


def test_summarize_candidates_reports_useful_wasted_and_missed() -> None:
    targets = np.empty((1, 255, 40, 8), dtype=np.uint8)
    targets[...] = np.arange(8, dtype=np.uint8)
    candidates = np.empty((1, 255, 40, 16), dtype=np.uint8)
    candidates[...] = np.arange(4, 20, dtype=np.uint8)
    metrics = summarize_candidates(candidates, targets, history_tokens=8)
    assert metrics["mean_useful_candidates"] == pytest.approx(4.0)
    assert metrics["mean_incorrect_candidates"] == pytest.approx(12.0)
    assert metrics["mean_missed_actual_experts"] == pytest.approx(4.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.25)
    assert metrics["fully_covered_rate"] == 0.0
    assert metrics["offload_layers_1_38"]["recall"] == pytest.approx(0.5)
    assert metrics["cold_start"]["samples"] == 7 * 40
    assert metrics["steady_state"]["samples"] == 248 * 40


def test_duplicate_candidates_are_rejected() -> None:
    targets = np.zeros((1, 255, 40, 8), dtype=np.uint8)
    candidates = np.zeros((1, 255, 40, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_candidates(candidates, targets, history_tokens=8)


def test_candidate_prefixes_use_the_same_ranked_prediction() -> None:
    targets = np.empty((1, 255, 40, 8), dtype=np.uint8)
    targets[...] = np.arange(8, dtype=np.uint8)
    candidates = np.empty((1, 255, 40, 32), dtype=np.uint8)
    candidates[...] = np.arange(32, dtype=np.uint8)
    metrics = summarize_candidate_prefixes(
        candidates,
        targets,
        candidate_counts=[8, 16, 24, 32],
        history_tokens=8,
    )
    assert metrics["8"]["recall"] == 1.0
    assert metrics["16"]["recall"] == 1.0
    assert metrics["24"]["recall"] == 1.0
    assert metrics["32"]["recall"] == 1.0
    assert metrics["8"]["precision"] == 1.0
    assert metrics["16"]["precision"] == 0.5
    assert metrics["24"]["precision"] == pytest.approx(1 / 3)
    assert metrics["32"]["precision"] == 0.25
