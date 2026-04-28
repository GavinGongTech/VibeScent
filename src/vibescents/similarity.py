from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms


def cosine_similarity_matrix(
    left: np.ndarray, right: np.ndarray | None = None
) -> np.ndarray:
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


def mmr_select(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    lambda_param: float = 0.5,
    top_k: int = 3,
) -> np.ndarray:
    if len(candidate_indices) <= top_k:
        return candidate_indices

    top_k = min(top_k, len(candidate_indices))
    candidates = corpus_emb[candidate_indices]
    sim_to_query = candidates @ query_emb
    candidate_sim_matrix = candidates @ candidates.T

    selected_indices = []
    remaining_indices = list(range(len(candidate_indices)))

    first_pick = np.argmax(sim_to_query)
    selected_indices.append(first_pick)
    remaining_indices.pop(first_pick)

    while len(selected_indices) < top_k:
        remaining_relevance = sim_to_query[remaining_indices]
        remaining_sim_to_selected = np.max(
            candidate_sim_matrix[remaining_indices][:, selected_indices], axis=1
        )
        mmr_scores = (
            lambda_param * remaining_relevance
            - (1 - lambda_param) * remaining_sim_to_selected
        )

        best_idx_in_remaining = np.argmax(mmr_scores)
        best_candidate = remaining_indices[best_idx_in_remaining]

        selected_indices.append(best_candidate)
        remaining_indices.pop(best_idx_in_remaining)

    return candidate_indices[selected_indices]
