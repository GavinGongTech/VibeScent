from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class Settings:
    # Qwen3-VL unified stack — local GPU, zero API keys required
    multimodal_embedding_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    # CPU fallback embedder (sentence-transformers, no GPU needed)
    text_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    reranker_model: str = "Qwen/Qwen3-VL-Reranker-8B"

    # LLM enrichment — any HuggingFace instruction-tuned model works with xgrammar guided decoding
    # Alternatives: "Qwen/Qwen3-14B", "google/gemma-3-12b-it", "google/gemma-3-27b-it"
    enrichment_model: str = "Qwen/Qwen3-8B"

    embedding_dimensions: int = 384
    rerank_top_k: int = 5   # candidates sent to Nvidia NIM reranker (fewer = faster)
    retrieve_top_k: int = 10  # candidates retrieved before reranking

    # Populated by re_embed_corpus_cpu.py after MiniLM re-embedding
    corpus_embeddings_path: str = str(
        DEFAULT_ARTIFACTS_DIR / "minilm_corpus" / "embeddings.npy"
    )
    corpus_metadata_path: str = str(PROJECT_ROOT / "data" / "vibescent_enriched.csv")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            multimodal_embedding_model=os.environ.get(
                "MULTIMODAL_EMBEDDING_MODEL",
                "Qwen/Qwen3-VL-Embedding-8B",
            ),
            text_embedding_model=os.environ.get(
                "TEXT_EMBEDDING_MODEL",
                "nomic-ai/nomic-embed-text-v1.5",
            ),
            reranker_model=os.environ.get(
                "RERANKER_MODEL",
                "Qwen/Qwen3-VL-Reranker-8B",
            ),
            enrichment_model=os.environ.get(
                "ENRICHMENT_MODEL",
                "Qwen/Qwen3-8B",
            ),
            corpus_embeddings_path=os.environ.get(
                "CORPUS_EMBEDDINGS_PATH",
                str(DEFAULT_ARTIFACTS_DIR / "minilm_corpus" / "embeddings.npy"),
            ),
            corpus_metadata_path=os.environ.get(
                "CORPUS_METADATA_PATH",
                str(PROJECT_ROOT / "data" / "vibescent_enriched.csv"),
            ),
        )
