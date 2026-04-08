from __future__ import annotations

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
        response = self._client.models.embed_content(
            model=model or self.settings.text_embedding_model,
            contents=text_list,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality or self.settings.embedding_dimensions,
            ),
        )
        return self._extract_matrix(response)

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
