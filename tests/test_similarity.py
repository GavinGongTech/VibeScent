from __future__ import annotations

import numpy as np

from vibescents.similarity import (
    cosine_similarity_matrix,
    normalize_rows,
    top_k_indices,
    weighted_sum,
)


def test_normalize_rows_handles_zero_rows() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = normalize_rows(matrix)
    assert np.allclose(normalized[0], np.array([0.6, 0.8], dtype=np.float32))
    assert np.allclose(normalized[1], np.array([0.0, 0.0], dtype=np.float32))


def test_cosine_similarity_matrix_matches_expected_shape() -> None:
    left = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    right = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    similarity = cosine_similarity_matrix(left, right)
    assert similarity.shape == (2, 2)
    assert similarity[0, 0] > similarity[0, 1]


def test_top_k_indices_returns_descending_order() -> None:
    scores = np.array([0.1, 0.8, 0.4, 0.9], dtype=np.float32)
    indices = top_k_indices(scores, 3)
    assert list(indices) == [3, 1, 2]


def test_weighted_sum_combines_named_signals() -> None:
    text = np.array([0.3, 0.7], dtype=np.float32)
    image = np.array([0.8, 0.2], dtype=np.float32)
    combined = weighted_sum(
        {
            "text": (text, 0.5),
            "image": (image, 0.5),
        }
    )
    assert np.allclose(combined, np.array([0.55, 0.45], dtype=np.float32))
