from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np

from vibescents.io_utils import guess_mime_type
from vibescents.settings import Settings


class GeminiEmbedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        if not self.settings.api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY before calling the Gemini API.")

        from google import genai

        self._client = genai.Client(api_key=self.settings.api_key)

    _MAX_BATCH_SIZE = 100

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        model: str | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int | None = None,
    ) -> np.ndarray:
        from google.genai import types

        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)

        all_rows: list[np.ndarray] = []
        n_batches = (len(text_list) + self._MAX_BATCH_SIZE - 1) // self._MAX_BATCH_SIZE
        for i, start in enumerate(range(0, len(text_list), self._MAX_BATCH_SIZE)):
            batch = text_list[start : start + self._MAX_BATCH_SIZE]
            for attempt in range(5):
                try:
                    response = self._client.models.embed_content(
                        model=model or self.settings.text_embedding_model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=output_dimensionality or self.settings.embedding_dimensions,
                        ),
                    )
                    all_rows.append(self._extract_matrix(response))
                    break
                except Exception as e:
                    if attempt < 4 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                        wait = min(30 * (attempt + 1), 60)
                        print(f"  Rate limited on batch {i+1}, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            if n_batches > 1 and i < n_batches - 1:
                print(f"  Embedded batch {i+1}/{n_batches} ({start + len(batch)}/{len(text_list)} texts)")
                time.sleep(2)
        return np.vstack(all_rows)

    def embed_multimodal_query(
        self,
        *,
        text: str,
        image_path: str | Path | None = None,
        task_type: str = "RETRIEVAL_QUERY",
        output_dimensionality: int | None = None,
    ) -> np.ndarray:
        from google.genai import types

        parts = [types.Part.from_text(text=text)]
        if image_path is not None:
            image_file = Path(image_path)
            parts.append(
                types.Part.from_bytes(
                    data=image_file.read_bytes(),
                    mime_type=guess_mime_type(image_file),
                )
            )
        content = types.Content(role="user", parts=parts)
        response = self._client.models.embed_content(
            model=self.settings.multimodal_embedding_model,
            contents=[content],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality or self.settings.embedding_dimensions,
            ),
        )
        return self._extract_matrix(response)

    def embed_multimodal_documents(
        self,
        texts: Iterable[str],
        *,
        output_dimensionality: int | None = None,
    ) -> np.ndarray:
        return self.embed_texts(
            texts,
            model=self.settings.multimodal_embedding_model,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=output_dimensionality or self.settings.embedding_dimensions,
        )

    @staticmethod
    def _extract_matrix(response: object) -> np.ndarray:
        if hasattr(response, "embeddings") and response.embeddings:
            rows = [np.asarray(embedding.values, dtype=np.float32) for embedding in response.embeddings]
            return np.vstack(rows)
        if hasattr(response, "embedding") and response.embedding:
            return np.asarray([response.embedding.values], dtype=np.float32)
        raise ValueError("Gemini embedding response did not contain vectors.")
