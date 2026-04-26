from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibescents.engine import VibeScoreEngine, _parse_notes, _str_or_none
from vibescents.image_scoring import ImageHeadProbabilities
from vibescents.schemas import ContextInput


# ---- module-level helpers ----

def test_parse_notes_basic() -> None:
    result = _parse_notes("rose, jasmine", "amber, musk", "sandalwood")
    assert "rose" in result
    assert "jasmine" in result
    assert "sandalwood" in result


def test_parse_notes_skips_nan() -> None:
    result = _parse_notes(None, "nan", "", "cedar")
    assert result == ["cedar"]


def test_parse_notes_limits_to_8() -> None:
    long_notes = ", ".join(f"note{i}" for i in range(20))
    result = _parse_notes(long_notes)
    assert len(result) <= 8


def test_parse_notes_empty_all() -> None:
    assert _parse_notes(None, "", "nan") == []


def test_str_or_none_valid_string() -> None:
    assert _str_or_none("hello") == "hello"


def test_str_or_none_strips_whitespace() -> None:
    assert _str_or_none("  hello  ") == "hello"


def test_str_or_none_nan_returns_none() -> None:
    assert _str_or_none("nan") is None


def test_str_or_none_none_keyword_returns_none() -> None:
    assert _str_or_none("None") is None


def test_str_or_none_empty_returns_none() -> None:
    assert _str_or_none("") is None
    assert _str_or_none(None) is None


# ---- VibeScoreEngine construction ----

def _make_engine(n: int = 10) -> tuple[VibeScoreEngine, pd.DataFrame]:
    embeddings = np.random.rand(n, 64).astype(np.float32)
    df = pd.DataFrame({
        "name": [f"Fragrance {i}" for i in range(n)],
        "brand": ["Brand"] * n,
        "formality": np.linspace(0.0, 1.0, n),
        "day_night": np.linspace(0.0, 1.0, n),
        "fresh_warm": np.linspace(0.0, 1.0, n),
        "likely_season": ["spring", "summer", "fall", "winter", "all-season",
                          "spring", "summer", "fall", "winter", "all-season"][:n],
        "likely_occasion": ["Evening"] * n,
        "vibe_sentence": ["A nice scent."] * n,
        "top_notes": ["rose"] * n,
        "middle_notes": ["amber"] * n,
        "base_notes": ["musk"] * n,
    })
    return VibeScoreEngine(corpus_embeddings=embeddings, corpus_df=df), df


def test_engine_init_normalises_embeddings() -> None:
    engine, _ = _make_engine(5)
    norms = np.linalg.norm(engine._corpus_emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---- _fuse ----

def test_fuse_structured_only() -> None:
    engine, _ = _make_engine(5)
    s = np.random.rand(5).astype(np.float32)
    fused = engine._fuse(None, None, None, s)
    assert np.array_equal(fused, s)


def test_fuse_text_and_structured() -> None:
    engine, _ = _make_engine(5)
    t = np.random.rand(5).astype(np.float32)
    s = np.random.rand(5).astype(np.float32)
    fused = engine._fuse(t, None, None, s)
    assert fused.shape == (5,)


def test_fuse_text_image_structured() -> None:
    engine, _ = _make_engine(5)
    t = np.random.rand(5).astype(np.float32)
    img = np.random.rand(5).astype(np.float32)
    s = np.random.rand(5).astype(np.float32)
    fused = engine._fuse(t, None, img, s)
    assert fused.shape == (5,)


def test_fuse_all_channels() -> None:
    engine, _ = _make_engine(5)
    t = np.random.rand(5).astype(np.float32)
    m = np.random.rand(5).astype(np.float32)
    img = np.random.rand(5).astype(np.float32)
    s = np.random.rand(5).astype(np.float32)
    fused = engine._fuse(t, m, img, s)
    assert fused.shape == (5,)


def test_fuse_image_and_structured_fallback() -> None:
    engine, _ = _make_engine(5)
    img = np.random.rand(5).astype(np.float32)
    s = np.random.rand(5).astype(np.float32)
    fused = engine._fuse(None, None, img, s)
    assert fused.shape == (5,)


# ---- _vectorised_image_scores ----

def test_vectorised_image_scores_shape() -> None:
    engine, _ = _make_engine(10)
    probs = ImageHeadProbabilities(
        formal=np.array([0.3, 0.5, 0.2]),
        season=np.array([0.4, 0.3, 0.2, 0.1]),
        time=np.array([0.6, 0.4]),
    )
    scores = engine._vectorised_image_scores(probs)
    assert scores.shape == (10,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_vectorised_image_scores_handles_empty_season() -> None:
    n = 5
    embeddings = np.random.rand(n, 32).astype(np.float32)
    df = pd.DataFrame({
        "name": [f"F{i}" for i in range(n)],
        "brand": ["B"] * n,
        "formality": [0.5] * n,
        "day_night": [0.5] * n,
        "fresh_warm": [0.5] * n,
        "likely_season": ["", "nan", "all-season", "spring", "summer"],
        "likely_occasion": ["Evening"] * n,
        "vibe_sentence": ["x"] * n,
        "top_notes": ["rose"] * n,
        "middle_notes": ["amber"] * n,
        "base_notes": ["musk"] * n,
    })
    engine = VibeScoreEngine(corpus_embeddings=embeddings, corpus_df=df)
    probs = ImageHeadProbabilities(
        formal=np.array([0.3, 0.5, 0.2]),
        season=np.array([0.4, 0.3, 0.2, 0.1]),
        time=np.array([0.6, 0.4]),
    )
    scores = engine._vectorised_image_scores(probs)
    assert not np.any(np.isnan(scores))


# ---- _build_response ----

def test_build_response_top_3() -> None:
    engine, _ = _make_engine(10)
    fused = np.random.rand(10).astype(np.float32)
    top3 = np.argsort(fused)[-3:][::-1]
    resp = engine._build_response(top3, fused, ContextInput())
    assert len(resp.recommendations) == 3
    assert resp.recommendations[0].rank == 1
    assert resp.recommendations[2].rank == 3


def test_build_response_fallback_occasion() -> None:
    engine, _ = _make_engine(3)
    fused = np.array([0.9, 0.7, 0.5], dtype=np.float32)
    top3 = np.array([0, 1, 2])
    ctx = ContextInput(eventType="Gala")
    resp = engine._build_response(top3, fused, ctx)
    # all rows have "Evening" as likely_occasion in _make_engine
    assert all(r.occasion for r in resp.recommendations)


# ---- _get_embedder sentinel ----

def test_get_embedder_marks_unavailable_on_failure() -> None:
    engine, _ = _make_engine(3)
    with pytest.MonkeyPatch().context() as mp:
        import vibescents.embeddings as emb_mod
        mp.setattr(emb_mod, "Qwen3VLMultimodalEmbedder", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no GPU")))
        result = engine._get_embedder()
    assert result is None
    assert engine._embedder is VibeScoreEngine._EMBEDDER_UNAVAILABLE
