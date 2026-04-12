from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    voyage_api_key: str | None = None
    text_embedding_model: str = "voyage-3-large"
    multimodal_embedding_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    reranker_model: str = "gemini-3.1-pro-preview"
    judge_model: str = "gemini-2.5-pro"
    embedding_dimensions: int = 1024
    rerank_top_k: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
