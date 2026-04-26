from __future__ import annotations

import numpy as np
import pandas as pd

from vibescents.schemas import ContextInput

# Target values per option for each enriched-CSV dimension
_FORMALITY_TARGETS: dict[str, float] = {
    "Gala": 0.90,
    "Date Night": 0.60,
    "Casual": 0.15,
    "Business": 0.75,
    "Wedding": 0.85,
    "Festival": 0.25,
}

_DAY_NIGHT_TARGETS: dict[str, float] = {
    "Morning": 0.10,
    "Afternoon": 0.30,
    "Evening": 0.70,
    "Night": 0.90,
}

_FRESH_WARM_TARGETS: dict[str, float] = {
    "Bold": 0.65,
    "Subtle": 0.35,
    "Fresh": 0.05,
    "Warm": 0.90,
    "Mysterious": 0.75,
}


def compute_structured_scores(ctx: ContextInput, df: pd.DataFrame) -> np.ndarray:
    """
    Score each row in df against the user context.
    Returns float32 array of shape (len(df),) with values in [0, 1].
    A score of 1.0 means perfect match; 0.0 means opposite.
    """
    n = len(df)
    scores = np.zeros(n, dtype=np.float32)
    weight_total = 0.0

    def _coerce(col: str) -> np.ndarray:
        return (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0.5)
            .values.astype(np.float32)
        )

    if ctx.eventType and ctx.eventType in _FORMALITY_TARGETS:
        target = _FORMALITY_TARGETS[ctx.eventType]
        scores += 1.0 - np.abs(_coerce("formality") - target)
        weight_total += 1.0

    if ctx.timeOfDay and ctx.timeOfDay in _DAY_NIGHT_TARGETS:
        target = _DAY_NIGHT_TARGETS[ctx.timeOfDay]
        scores += 1.0 - np.abs(_coerce("day_night") - target)
        weight_total += 1.0

    if ctx.mood and ctx.mood in _FRESH_WARM_TARGETS:
        target = _FRESH_WARM_TARGETS[ctx.mood]
        scores += 1.0 - np.abs(_coerce("fresh_warm") - target)
        weight_total += 1.0

    if weight_total == 0.0:
        return np.full(n, 0.5, dtype=np.float32)

    return scores / weight_total
