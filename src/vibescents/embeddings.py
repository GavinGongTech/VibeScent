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
        self, settings: Settings | None = None, *, load_in_8bit: bool = False
    ) -> None:
        self.settings = settings or Settings.from_env()
        import torch
        from vibescents.qwen3_vl_embedding import Qwen3VLEmbedder as _Inner

        _model_kwargs: dict = (
            {"load_in_8bit": True} if load_in_8bit else {"torch_dtype": torch.bfloat16}
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
        from tqdm.auto import tqdm

        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        items = [{"text": t} for t in text_list]
        rows: list[np.ndarray] = []
        n = len(items)
        with tqdm(total=n, desc="Embedding multimodal documents") as pbar:
            for i in range(0, n, self._BATCH_SIZE):
                batch = items[i : i + self._BATCH_SIZE]
                result = self._embedder.process(batch)
                rows.append(result.cpu().float().numpy())
                pbar.update(len(batch))
        return np.vstack(rows)
