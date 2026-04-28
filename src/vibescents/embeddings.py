from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from vibescents.settings import Settings


class Qwen3VLMultimodalEmbedder:
    """Multimodal embedder backed by Qwen3-VL-Embedding-8B (local GPU, no API key).

    MMEB-V2 #1 at 77.8% — embeds images and text in the same vector space natively.
    Requires: pip install torch transformers>=4.57.0 qwen-vl-utils>=0.0.14 accelerate
    GPU: ~16 GB VRAM in bfloat16 (fits A100 40 GB or Blackwell 6000 80 GB).
    """

    _BATCH_SIZE = 8

    def __init__(
        self, settings: Settings | None = None, *, load_in_8bit: bool = False, load_in_4bit: bool = False
    ) -> None:
        self.settings = settings or Settings.from_env()
        import torch
        from vibescents.qwen3_vl_embedding import Qwen3VLEmbedder as _Inner

        _model_kwargs: dict = {}
        if load_in_4bit:
            _model_kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            _model_kwargs["load_in_8bit"] = True
        else:
            _model_kwargs["torch_dtype"] = torch.bfloat16
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
        *,
        batch_size: int | None = None,
        **_kwargs,
    ) -> np.ndarray:
        from tqdm.auto import tqdm

        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        items = [{"text": t} for t in text_list]
        rows: list[np.ndarray] = []
        n = len(items)
        b = batch_size or self._BATCH_SIZE
        with tqdm(total=n, desc="Embedding multimodal documents") as pbar:
            for i in range(0, n, b):
                batch = items[i : i + b]
                result = self._embedder.process(batch)
                rows.append(result.cpu().float().numpy())
                pbar.update(len(batch))
        return np.vstack(rows)


class SentenceTransformerEmbedder:
    """CPU-friendly text embedder via sentence-transformers.

    Default: nomic-ai/nomic-embed-text-v1.5 (137M params, 768-dim, Apache 2.0).
    Fallback when Qwen3-VL-Embedding-8B is unavailable (no GPU).
    Note: text-only — outfit images are not embedded, multimodal channel is disabled.
    Install: pip install sentence-transformers
    """

    _QUERY_PREFIX = "search_query: "
    _DOC_PREFIX = "search_document: "
    _NOMIC_MODELS = {"nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1"}

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        *,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._needs_prefix = model_name in self._NOMIC_MODELS
        self._model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device
        )

    def embed_multimodal_documents(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        **_kwargs,
    ) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        if self._needs_prefix:
            text_list = [self._DOC_PREFIX + t for t in text_list]
        return self._model.encode(
            text_list,
            batch_size=batch_size or 256,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def embed_multimodal_query(
        self, *, text: str, image_path=None, **_kwargs
    ) -> np.ndarray:
        raise NotImplementedError(
            "SentenceTransformerEmbedder is text-only. "
            "Multimodal channel will be skipped automatically."
        )
