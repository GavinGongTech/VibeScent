from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping

import numpy as np

DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "text": 0.275,
    "image": 0.450,
    "structured": 0.275,
}


@dataclass(frozen=True)
class GridSearchResult:
    weights: dict[str, float]
    metric: float
    scores: np.ndarray


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    values = scores.astype(np.float32, copy=False)
    if values.ndim != 1:
        raise ValueError("scores must be a 1-D vector.")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum == minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def normalize_signal_map(score_map: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not score_map:
        raise ValueError("score_map must not be empty.")
    lengths = {scores.shape[0] for scores in score_map.values()}
    if len(lengths) != 1:
        raise ValueError("All score vectors must have the same length.")
    return {name: min_max_normalize(scores) for name, scores in score_map.items()}


def fuse_scores(
    score_map: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    normalized = normalize_signal_map(score_map)
    selected_weights = dict(weights or DEFAULT_FUSION_WEIGHTS)
    if set(normalized) != set(selected_weights):
        missing = sorted(set(normalized) - set(selected_weights))
        extra = sorted(set(selected_weights) - set(normalized))
        raise ValueError(
            f"Weight keys must match score keys (missing={missing}, extra={extra})."
        )
    weight_total = sum(float(weight) for weight in selected_weights.values())
    if weight_total <= 0:
        raise ValueError("Weight sum must be positive.")
    if not np.isclose(weight_total, 1.0):
        raise ValueError(f"Weight sum must equal 1.0, got {weight_total:.6f}.")
    result = np.zeros_like(next(iter(normalized.values())))
    for name, scores in normalized.items():
        result += scores * float(selected_weights[name])
    return result


def build_weight_grid(
    *,
    channels: tuple[str, ...] = ("text", "multimodal", "image", "structured"),
    step: float = 0.05,
) -> list[dict[str, float]]:
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")
    bucket_count = int(round(1.0 / step))
    if not np.isclose(bucket_count * step, 1.0):
        raise ValueError("step must partition 1.0 evenly.")
    grid: list[dict[str, float]] = []
    for assignment in product(range(bucket_count + 1), repeat=len(channels)):
        if sum(assignment) != bucket_count:
            continue
        grid.append(
            {
                channel: bucket * step
                for channel, bucket in zip(channels, assignment, strict=True)
            }
        )
    return grid


def _fuse_normalized(
    normalized_map: dict[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Weighted sum of already-normalized signal vectors (no re-normalization)."""
    result = np.zeros_like(next(iter(normalized_map.values())))
    for name, scores in normalized_map.items():
        result += scores * float(weights[name])
    return result


def grid_search_weights(
    score_map: Mapping[str, np.ndarray],
    *,
    scorer: Callable[[np.ndarray], float],
    weight_grid: Iterable[Mapping[str, float]],
) -> GridSearchResult:
    normalized = normalize_signal_map(
        score_map
    )  # normalize once across all grid iterations
    best: GridSearchResult | None = None
    for candidate_weights in weight_grid:
        fused = _fuse_normalized(normalized, candidate_weights)
        metric = float(scorer(fused))
        if best is None or metric > best.metric:
            best = GridSearchResult(
                weights=dict(candidate_weights), metric=metric, scores=fused
            )
    if best is None:
        raise ValueError("weight_grid must not be empty.")
    return best
