from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibescents.io_utils import (
    dump_json,
    ensure_dir,
    guess_mime_type,
    load_dataframe,
    load_embeddings,
    load_json,
    save_dataframe,
    save_embeddings,
)


def test_ensure_dir_creates_nested(tmp_path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.is_dir()


def test_ensure_dir_idempotent(tmp_path) -> None:
    ensure_dir(tmp_path)
    ensure_dir(tmp_path)  # should not raise


def test_dump_and_load_json_roundtrip(tmp_path) -> None:
    data = {"key": "val", "num": 42, "nested": {"list": [1, 2, 3]}}
    path = tmp_path / "test.json"
    dump_json(path, data)
    loaded = load_json(path)
    assert loaded == data


def test_save_and_load_dataframe(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data.csv"
    save_dataframe(path, df)
    loaded = load_dataframe(path)
    assert list(loaded["a"]) == [1, 2, 3]
    assert list(loaded["b"]) == ["x", "y", "z"]


def test_save_and_load_embeddings(tmp_path) -> None:
    matrix = np.random.rand(8, 32).astype(np.float32)
    path = tmp_path / "emb.npy"
    save_embeddings(path, matrix)
    loaded = load_embeddings(path)
    assert loaded.shape == (8, 32)
    assert np.allclose(loaded, matrix)


def test_guess_mime_type_jpeg(tmp_path) -> None:
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"fake")
    assert guess_mime_type(p) == "image/jpeg"


def test_guess_mime_type_png(tmp_path) -> None:
    p = tmp_path / "img.png"
    p.write_bytes(b"fake")
    assert guess_mime_type(p) == "image/png"


def test_guess_mime_type_webp(tmp_path) -> None:
    p = tmp_path / "img.webp"
    p.write_bytes(b"fake")
    assert guess_mime_type(p) == "image/webp"


def test_guess_mime_type_unknown_raises(tmp_path) -> None:
    p = tmp_path / "file.xyzxyz"
    p.write_bytes(b"fake")
    with pytest.raises(ValueError, match="MIME"):
        guess_mime_type(p)
