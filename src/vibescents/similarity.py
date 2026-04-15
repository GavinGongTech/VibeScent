from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    right_matrix = left if right is None else right
    return normalize_rows(left) @ normalize_rows(right_matrix).T


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, scores.shape[0])
    unsorted = np.argpartition(scores, -k)[-k:]
    return unsorted[np.argsort(scores[unsorted])[::-1]]


def weighted_sum(score_map: dict[str, tuple[np.ndarray, float]]) -> np.ndarray:
    if not score_map:
        raise ValueError("score_map must not be empty")
    result = None
    for scores, weight in score_map.values():
        weighted = scores * weight
        result = weighted if result is None else result + weighted
    return result


def majority_vote(values: Sequence[str]) -> str:
    return Counter(values).most_common(1)[0][0]


def frequent_items(items: Iterable[Iterable[str]], min_frequency: int = 2) -> list[str]:
    counter = Counter(item for group in items for item in group)
    return sorted(item for item, count in counter.items() if count >= min_frequency)
