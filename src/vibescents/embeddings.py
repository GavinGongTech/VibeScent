from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np

from vibescents.settings import Settings


class VoyageEmbedder:
    """Text embedder backed by Voyage AI (voyage-3-large, MTEB English #1 at 68.32).

    Requires: pip install voyageai
    API key: VOYAGE_API_KEY env var.
    Dimensions: 1024 (default), 256/512/2048 also supported.
    """

    _MAX_BATCH_SIZE = 128

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        key = self.settings.voyage_api_key
        if not key:
            raise ValueError("Set VOYAGE_API_KEY before calling the Voyage AI API.")
        self._key = key

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        model: str | None = None,
        input_type: str = "document",
        output_dimension: int | None = None,
    ) -> np.ndarray:
        import requests

        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)

        m = model or self.settings.text_embedding_model
        dim = output_dimension or self.settings.embedding_dimensions
        all_rows: list[np.ndarray] = []
        n_batches = (len(text_list) + self._MAX_BATCH_SIZE - 1) // self._MAX_BATCH_SIZE
        for i, start in enumerate(range(0, len(text_list), self._MAX_BATCH_SIZE)):
            batch = text_list[start : start + self._MAX_BATCH_SIZE]
            for attempt in range(5):
                try:
                    payload = {
                        "input": batch,
                        "model": m,
                        "input_type": input_type,
                    }
                    if dim:
                        payload["output_dimension"] = dim

                    response = requests.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Sort data by index just in case
                    embeddings = [
                        item["embedding"]
                        for item in sorted(
                            data["data"], key=lambda x: x.get("index", 0)
                        )
                    ]
                    all_rows.append(np.array(embeddings, dtype=np.float32))
                    break
                except Exception as e:
                    if attempt < 4 and ("429" in str(e) or "rate" in str(e).lower()):
                        wait = min(30 * (attempt + 1), 60)
                        print(f"  Rate limited on batch {i + 1}, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            if n_batches > 1 and i < n_batches - 1:
                print(
                    f"  Embedded batch {i + 1}/{n_batches} ({start + len(batch)}/{len(text_list)} texts)"
                )
                time.sleep(1)
        return np.vstack(all_rows)


class GeminiEmbedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        if not self.settings.api_key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY before calling the Gemini API."
            )

        from google import genai

        self._client = genai.Client(api_key=self.settings.api_key)

    _MAX_BATCH_SIZE = 100

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        model: str | None = None,
        task_type: str | None = None,
        input_type: str | None = None,
        output_dimensionality: int | None = None,
    ) -> np.ndarray:
        from google.genai import types

        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)

        if not task_type:
            if input_type == "query":
                task_type = "RETRIEVAL_QUERY"
            else:
                task_type = "RETRIEVAL_DOCUMENT"

        all_rows: list[np.ndarray] = []
        n_batches = (len(text_list) + self._MAX_BATCH_SIZE - 1) // self._MAX_BATCH_SIZE
        for i, start in enumerate(range(0, len(text_list), self._MAX_BATCH_SIZE)):
            batch = text_list[start : start + self._MAX_BATCH_SIZE]
            for attempt in range(100):
                try:
                    response = self._client.models.embed_content(
                        model=model or self.settings.text_embedding_model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=output_dimensionality
                            or self.settings.embedding_dimensions,
                        ),
                    )
                    all_rows.append(self._extract_matrix(response))
                    break
                except Exception as e:
                    if attempt < 99 and (
                        "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                    ):
                        import re

                        delay = 60.0
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e))
                        if match:
                            delay = float(match.group(1)) + 1.0
                        else:
                            delay = min(300.0, 2 ** (attempt + 1))
                        print(
                            f"  Rate limited on batch {i + 1}, waiting {delay:.1f}s... (attempt {attempt + 1})"
                        )
                        time.sleep(delay)
                    else:
                        raise
            if n_batches > 1 and i < n_batches - 1:
                print(
                    f"  Embedded batch {i + 1}/{n_batches} ({start + len(batch)}/{len(text_list)} texts)"
                )
                time.sleep(2)
        return np.vstack(all_rows)

    @staticmethod
    def _extract_matrix(response: object) -> np.ndarray:
        if hasattr(response, "embeddings") and response.embeddings:
            rows = [
                np.asarray(embedding.values, dtype=np.float32)
                for embedding in response.embeddings
            ]
            return np.vstack(rows)
        if hasattr(response, "embedding") and response.embedding:
            return np.asarray([response.embedding.values], dtype=np.float32)
        raise ValueError("Gemini embedding response did not contain vectors.")


class Qwen3VLMultimodalEmbedder:
    """Multimodal embedder backed by Qwen3-VL-Embedding-8B (local GPU, MMEB-V2 #1 at 77.8).

    Requires: pip install torch transformers>=4.57.0 qwen-vl-utils>=0.0.14 accelerate
    GPU: ~16 GB VRAM in float16 (fits Colab Pro A100).
    """

    _BATCH_SIZE = 8

    def __init__(
        self, settings: Settings | None = None, *, load_in_8bit: bool = False
    ) -> None:
        self.settings = settings or Settings.from_env()
        import torch
        from vibescents.qwen3_vl_embedding import Qwen3VLEmbedder as _Inner

        _model_kwargs: dict = (
            {"load_in_8bit": True} if load_in_8bit else {"torch_dtype": torch.float16}
        )
        self._embedder = _Inner(
            model_name_or_path=self.settings.multimodal_embedding_model,
            **_model_kwargs,
        )

    def embed_multimodal_query(
        self,
        *,
        text: str,
        image_path: str | Path | None = None,
        **_kwargs,
    ) -> np.ndarray:
        item: dict = {"text": text}
        if image_path is not None:
            item["image"] = str(Path(image_path).resolve())
        result = self._embedder.process([item])
        return result.cpu().float().numpy()

    def embed_multimodal_documents(
        self,
        texts: Iterable[str],
        **_kwargs,
    ) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        items = [{"text": t} for t in text_list]
        rows: list[np.ndarray] = []
        n = len(items)
        for i in range(0, n, self._BATCH_SIZE):
            batch = items[i : i + self._BATCH_SIZE]
            result = self._embedder.process(batch)
            rows.append(result.cpu().float().numpy())
            if i + self._BATCH_SIZE < n:
                print(f"  Embedded {i + len(batch)}/{n} documents")
        return np.vstack(rows)
