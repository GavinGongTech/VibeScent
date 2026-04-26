from __future__ import annotations

from vibescents.settings import Settings


def test_from_env_returns_settings() -> None:
    s = Settings.from_env()
    assert isinstance(s, Settings)


def test_default_embedding_dimensions() -> None:
    s = Settings()
    assert s.embedding_dimensions == 1024


def test_default_top_k_values() -> None:
    s = Settings()
    assert s.rerank_top_k == 10
    assert s.retrieve_top_k == 20


def test_corpus_paths_are_strings() -> None:
    s = Settings()
    assert isinstance(s.corpus_embeddings_path, str)
    assert isinstance(s.corpus_metadata_path, str)
    assert s.corpus_embeddings_path.endswith("embeddings.npy")
    assert s.corpus_metadata_path.endswith("vibescent_enriched.csv")
