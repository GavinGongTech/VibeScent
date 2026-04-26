from __future__ import annotations

import numpy as np
import pandas as pd

from vibescents.schemas import ContextInput
from vibescents.structured_scorer import compute_structured_scores


def _make_df(n: int = 5) -> pd.DataFrame:
    vals = np.linspace(0.0, 1.0, n)
    return pd.DataFrame({"formality": vals, "day_night": vals, "fresh_warm": vals})


def test_empty_context_returns_half(n: int = 6) -> None:
    df = _make_df(6)
    scores = compute_structured_scores(ContextInput(), df)
    assert scores.shape == (6,)
    assert np.allclose(scores, 0.5)


def test_event_type_only_in_range() -> None:
    df = _make_df(4)
    scores = compute_structured_scores(ContextInput(eventType="Gala"), df)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_time_only_in_range() -> None:
    df = _make_df(4)
    scores = compute_structured_scores(ContextInput(timeOfDay="Night"), df)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_mood_only_in_range() -> None:
    df = _make_df(4)
    scores = compute_structured_scores(ContextInput(mood="Warm"), df)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_all_fields_in_range() -> None:
    df = _make_df(5)
    ctx = ContextInput(eventType="Business", timeOfDay="Afternoon", mood="Bold")
    scores = compute_structured_scores(ctx, df)
    assert scores.shape == (5,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_unknown_event_type_falls_back_to_half() -> None:
    df = _make_df(3)
    scores = compute_structured_scores(ContextInput(eventType="Rave"), df)
    # only structured scoring weight is 0 so returns 0.5
    assert np.allclose(scores, 0.5)


def test_nan_values_filled_gracefully() -> None:
    df = pd.DataFrame(
        {
            "formality": [float("nan"), 0.5],
            "day_night": [0.3, float("nan")],
            "fresh_warm": [0.2, 0.8],
        }
    )
    ctx = ContextInput(eventType="Gala", timeOfDay="Evening", mood="Fresh")
    scores = compute_structured_scores(ctx, df)
    assert not np.any(np.isnan(scores))


def test_perfect_match_gives_near_one() -> None:
    df = pd.DataFrame({"formality": [0.90], "day_night": [0.90], "fresh_warm": [0.05]})
    ctx = ContextInput(eventType="Gala", timeOfDay="Night", mood="Fresh")
    scores = compute_structured_scores(ctx, df)
    assert scores[0] > 0.85
