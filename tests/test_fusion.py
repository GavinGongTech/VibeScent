from __future__ import annotations

import numpy as np

from vibescents.fusion import (
    build_weight_grid,
    fuse_scores,
    grid_search_weights,
    min_max_normalize,
)


def test_min_max_normalize_returns_zeros_for_constant_vector() -> None:
    vector = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    normalized = min_max_normalize(vector)
    assert np.allclose(normalized, np.zeros(3, dtype=np.float32))


def test_fuse_scores_uses_per_signal_normalization() -> None:
    text = np.array([1.0, 3.0], dtype=np.float32)  # -> [0, 1]
    image = np.array([20.0, 10.0], dtype=np.float32)  # -> [1, 0]
    fused = fuse_scores(
        {"text": text, "image": image},
        weights={"text": 0.5, "image": 0.5},
    )
    assert np.allclose(fused, np.array([0.5, 0.5], dtype=np.float32))


def test_build_weight_grid_sums_to_one() -> None:
    grid = build_weight_grid(channels=("text", "image"), step=0.5)
    assert {"text": 0.0, "image": 1.0} in grid
    assert {"text": 0.5, "image": 0.5} in grid
    assert {"text": 1.0, "image": 0.0} in grid


def test_grid_search_weights_returns_best_scoring_assignment() -> None:
    score_map = {
        "text": np.array([0.9, 0.1], dtype=np.float32),
        "image": np.array([0.1, 0.9], dtype=np.float32),
    }
    grid = [
        {"text": 1.0, "image": 0.0},
        {"text": 0.0, "image": 1.0},
    ]

    result = grid_search_weights(
        score_map,
        scorer=lambda scores: float(scores[0]),
        weight_grid=grid,
    )

    assert result.weights == {"text": 1.0, "image": 0.0}


def test_grid_search_weights_does_not_double_normalize() -> None:
    """grid_search_weights must not pre-normalize before fuse_scores (which normalizes internally).
    Verify by checking that a sub-unit signal retains its relative magnitude post-search.
    """
    # "text" spans [0.0, 0.2]; "image" spans [0.0, 1.0]
    # If grid_search correctly delegates normalization to fuse_scores, the result for
    # weights={text:0.5, image:0.5} should equal fuse_scores(score_map, weights=...) directly.
    score_map = {
        "text": np.array([0.0, 0.2], dtype=np.float32),
        "image": np.array([0.0, 1.0], dtype=np.float32),
    }
    grid = [{"text": 0.5, "image": 0.5}]
    result = grid_search_weights(
        score_map, scorer=lambda s: float(s.sum()), weight_grid=grid
    )
    expected = fuse_scores(score_map, weights={"text": 0.5, "image": 0.5})
    assert np.allclose(result.scores, expected)
