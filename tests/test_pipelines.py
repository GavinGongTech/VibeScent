from __future__ import annotations

import numpy as np
import pytest

from vibescents.pipelines import load_karans_embeddings


def test_load_karans_embeddings_truncates_and_normalizes(tmp_path) -> None:
    matrix = np.array(
        [
            [3.0, 4.0, 0.0, 0.0],
            [1.0, 2.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    source = tmp_path / "embeddings.npy"
    np.save(source, matrix)

    loaded = load_karans_embeddings(source, output_dim=2)
    assert loaded.shape == (2, 2)
    assert np.allclose(np.linalg.norm(loaded, axis=1), np.ones(2, dtype=np.float32))


def test_load_karans_embeddings_raises_on_dim_expansion(tmp_path) -> None:
    matrix = np.zeros((3, 384), dtype=np.float32)
    source = tmp_path / "embeddings.npy"
    np.save(source, matrix)

    with pytest.raises(ValueError):
        load_karans_embeddings(source, output_dim=1024)
