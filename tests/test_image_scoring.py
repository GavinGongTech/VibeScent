from __future__ import annotations

import math

import numpy as np

from vibescents.image_scoring import (
    discretize_day_night,
    discretize_formality,
    image_negative_log_likelihood,
    score_candidate_pool,
    season_target_index,
)


def test_discretize_formality_and_time_buckets() -> None:
    assert discretize_formality(0.0) == 0
    assert discretize_formality(0.5) == 1
    assert discretize_formality(0.9) == 2
    assert discretize_day_night(0.49) == 0
    assert discretize_day_night(0.50) == 1


def test_season_target_index_uses_argmax_for_all_season() -> None:
    probs = np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32)
    assert season_target_index("all-season", probs) == 2
    assert season_target_index("winter", probs) == 3


def test_image_negative_log_likelihood_matches_expected_value() -> None:
    head_probabilities = {
        "formal": np.array([0.8, 0.1, 0.1], dtype=np.float32),
        "season": np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
        "time": np.array([0.9, 0.1], dtype=np.float32),
    }
    fragrance = {
        "formality": 0.2,
        "day_night": 0.2,
        "likely_season": "spring",
    }
    nll = image_negative_log_likelihood(head_probabilities, fragrance)
    expected = -(math.log(0.8) + math.log(0.7) + math.log(0.9))
    assert math.isclose(nll, expected, rel_tol=1e-6)


def test_score_candidate_pool_returns_unit_range_values() -> None:
    head_probabilities = {
        "formal": np.array([0.8, 0.1, 0.1], dtype=np.float32),
        "season": np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
        "time": np.array([0.9, 0.1], dtype=np.float32),
    }
    candidates = [
        {"formality": 0.2, "day_night": 0.2, "likely_season": "spring"},
        {"formality": 0.8, "day_night": 0.8, "likely_season": "winter"},
    ]
    scores = score_candidate_pool(head_probabilities, candidates)
    assert scores.shape == (2,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
