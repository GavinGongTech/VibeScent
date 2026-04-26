from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vibescents.embeddings import Qwen3VLMultimodalEmbedder
from vibescents.fusion import fuse_scores
from vibescents.image_scorer import SigLIP2ImageScorer
from vibescents.image_scoring import ImageHeadProbabilities
from vibescents.io_utils import load_embeddings
from vibescents.query import context_to_query_string
from vibescents.schemas import (
    ContextInput,
    FragranceRecommendation,
    RecommendRequest,
    RecommendResponse,
)
from vibescents.settings import Settings
from vibescents.similarity import (
    cosine_similarity_matrix,
    normalize_rows,
    top_k_indices,
)
from vibescents.structured_scorer import compute_structured_scores

logger = logging.getLogger(__name__)

_SEASON_INDEX: dict[str, int] = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}

_FULL_WEIGHTS = {"text": 0.30, "multimodal": 0.25, "image": 0.30, "structured": 0.15}
_NO_IMAGE_WEIGHTS = {"text": 0.40, "multimodal": 0.35, "structured": 0.25}
_NO_MULTI_WEIGHTS = {"text": 0.45, "image": 0.40, "structured": 0.15}
_TEXT_ONLY_WEIGHTS = {"text": 0.80, "structured": 0.20}


class VibeScoreEngine:
    """
    4-channel fusion + listwise LLM reranker recommendation engine — zero API keys required.

    Channels:
      text (0.30)       — Qwen3-VL-Embedding-8B text query vs corpus
      multimodal (0.25) — Qwen3-VL-Embedding-8B image+text query vs corpus
      image (0.30)      — SigLIP 2 zero-shot → vectorised NLL vs enriched-DF attributes
      structured (0.15) — arithmetic context match vs enriched-DF numeric columns

    No LLM reranker in this engine (use VibeScentEngine in week5 notebook for full reranking).
    Falls back to fusion top-3 silently if the rerank call fails.

    The corpus (35k rows) was embedded with Qwen3-Embedding-8B. Query vectors come
    from GeminiEmbedder. Cross-model retrieval is imperfect but functional at demo
    scale. The listwise reranker partially compensates for this embedding space mismatch.
    Run harsh_offline_pipeline.ipynb to generate corpus embeddings with Qwen3-VL.
    """

    def __init__(
        self,
        corpus_embeddings: np.ndarray,
        corpus_df: pd.DataFrame,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._corpus_emb = normalize_rows(corpus_embeddings.astype(np.float32))
        self._corpus_df = corpus_df.reset_index(drop=True)
        self._settings = settings or Settings.from_env()
        self._embedder: object = None  # Qwen3VLMultimodalEmbedder, lazy-loaded; _EMBEDDER_UNAVAILABLE sentinel if no GPU
        self._siglip: SigLIP2ImageScorer | None = None

    @classmethod
    def from_artifacts(
        cls,
        *,
        embeddings_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        settings: Settings | None = None,
    ) -> "VibeScoreEngine":
        s = settings or Settings.from_env()
        emb_path = Path(embeddings_path or s.corpus_embeddings_path)
        meta_path = Path(metadata_path or s.corpus_metadata_path)

        logger.info("Loading corpus embeddings from %s", emb_path)
        embeddings = load_embeddings(emb_path)
        logger.info("Corpus shape: %s", embeddings.shape)

        logger.info("Loading corpus metadata from %s", meta_path)
        df = pd.read_csv(meta_path, low_memory=False)

        if len(df) != len(embeddings):
            raise ValueError(
                f"Corpus mismatch: {len(embeddings)} embeddings vs {len(df)} metadata rows."
            )

        return cls(corpus_embeddings=embeddings, corpus_df=df, settings=s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self, *, request: RecommendRequest, image_bytes: bytes
    ) -> RecommendResponse:
        query_str = context_to_query_string(request.context)
        logger.info("Query: %r", query_str)

        embedder = self._get_embedder()

        # Text channel
        text_scores: np.ndarray | None = None
        if embedder is not None:
            try:
                q_emb = embedder.embed_multimodal_documents([query_str])
                q_emb = normalize_rows(q_emb.astype(np.float32))
                text_scores = cosine_similarity_matrix(q_emb, self._corpus_emb)[0]
            except Exception as exc:
                logger.warning("Text embedding failed, skipping text channel: %s", exc)

        # Multimodal channel — Qwen3-VL-Embedding-8B sees image + context text natively
        multi_scores: np.ndarray | None = None
        if embedder is not None and image_bytes:
            try:
                import os as _os
                import tempfile as _tmf

                with _tmf.NamedTemporaryFile(suffix=".jpg", delete=False) as _f:
                    _f.write(image_bytes)
                    _tmp = _f.name
                try:
                    mm_emb = embedder.embed_multimodal_query(
                        text=query_str, image_path=_tmp
                    )
                finally:
                    _os.unlink(_tmp)
                mm_emb = normalize_rows(mm_emb.astype(np.float32))
                multi_scores = cosine_similarity_matrix(mm_emb, self._corpus_emb)[0]
            except Exception as exc:
                logger.warning("Multimodal embedding failed, skipping channel: %s", exc)

        # Image channel — SigLIP 2 zero-shot
        image_scores: np.ndarray | None = None
        try:
            head_probs = self._get_siglip().score_image(image_bytes)
            image_scores = self._vectorised_image_scores(head_probs)
        except Exception as exc:
            logger.warning("SigLIP 2 scoring failed, skipping image channel: %s", exc)

        # Structured channel — arithmetic match, never fails
        structured_scores = compute_structured_scores(request.context, self._corpus_df)

        fused = self._fuse(text_scores, multi_scores, image_scores, structured_scores)
        top3_indices = top_k_indices(fused, 3)

        return self._build_response(top3_indices, fused, request.context)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _EMBEDDER_UNAVAILABLE = object()

    def _get_embedder(self) -> Qwen3VLMultimodalEmbedder | None:
        if self._embedder is self._EMBEDDER_UNAVAILABLE:
            return None
        if self._embedder is None:
            try:
                self._embedder = Qwen3VLMultimodalEmbedder(self._settings)
            except Exception as exc:
                logger.warning(
                    "Qwen3-VL embedder unavailable (no GPU?), text/multimodal channels disabled: %s",
                    exc,
                )
                self._embedder = self._EMBEDDER_UNAVAILABLE
                return None
        return self._embedder  # type: ignore[return-value]

    def _get_siglip(self) -> SigLIP2ImageScorer:
        if self._siglip is None:
            self._siglip = SigLIP2ImageScorer()
        return self._siglip

    def _vectorised_image_scores(self, probs: ImageHeadProbabilities) -> np.ndarray:
        """Score all corpus rows against SigLIP 2 probabilities using vectorised NLL."""
        df = self._corpus_df
        eps = 1e-8

        formality = pd.to_numeric(df["formality"], errors="coerce").fillna(0.5).values
        day_night = pd.to_numeric(df["day_night"], errors="coerce").fillna(0.5).values
        seasons = df["likely_season"].fillna("all-season").astype(str).values

        # Map each fragrance's continuous attributes → discrete class indices
        formal_targets = np.where(formality < 0.33, 0, np.where(formality < 0.67, 1, 2))
        time_targets = (day_night >= 0.5).astype(int)

        best_season = int(np.argmax(probs.season))
        season_targets = np.array(
            [
                _SEASON_INDEX.get(s.strip().lower(), best_season)
                if s.strip().lower() not in ("all-season", "nan", "")
                else best_season
                for s in seasons
            ],
            dtype=int,
        )

        # Look up the SigLIP 2 probability assigned to each fragrance's target class
        formal_p = np.clip(probs.formal[formal_targets], eps, 1.0)
        time_p = np.clip(probs.time[time_targets], eps, 1.0)
        season_p = np.clip(probs.season[season_targets], eps, 1.0)

        nll = -(np.log(formal_p) + np.log(season_p) + np.log(time_p))
        return np.exp(-nll).astype(np.float32)

    def _fuse(
        self,
        text: np.ndarray | None,
        multi: np.ndarray | None,
        image: np.ndarray | None,
        structured: np.ndarray,
    ) -> np.ndarray:
        if text is not None and multi is not None and image is not None:
            return fuse_scores(
                {
                    "text": text,
                    "multimodal": multi,
                    "image": image,
                    "structured": structured,
                },
                weights=_FULL_WEIGHTS,
            )
        if text is not None and multi is not None:
            return fuse_scores(
                {"text": text, "multimodal": multi, "structured": structured},
                weights=_NO_IMAGE_WEIGHTS,
            )
        if text is not None and image is not None:
            return fuse_scores(
                {"text": text, "image": image, "structured": structured},
                weights=_NO_MULTI_WEIGHTS,
            )
        if text is not None:
            return fuse_scores(
                {"text": text, "structured": structured}, weights=_TEXT_ONLY_WEIGHTS
            )
        if image is not None:
            return fuse_scores(
                {"image": image, "structured": structured},
                weights={"image": 0.70, "structured": 0.30},
            )
        return structured

    def _build_response(
        self,
        top_indices: np.ndarray,
        fused_scores: np.ndarray,
        ctx: ContextInput,
    ) -> RecommendResponse:
        df = self._corpus_df
        recs: list[FragranceRecommendation] = []

        for rank, idx in enumerate(top_indices, start=1):
            row = df.iloc[int(idx)]
            notes = _parse_notes(
                row.get("top_notes"), row.get("middle_notes"), row.get("base_notes")
            )
            occasion = (
                _str_or_none(row.get("likely_occasion")) or ctx.eventType or "Evening"
            )
            reasoning = (
                _str_or_none(row.get("vibe_sentence"))
                or "A distinctive fragrance selected for your look."
            )
            recs.append(
                FragranceRecommendation(
                    rank=rank,
                    name=_str_or_none(row.get("name")) or "Unknown",
                    house=_str_or_none(row.get("brand")) or "Unknown",
                    score=round(float(fused_scores[int(idx)]), 3),
                    notes=notes,
                    reasoning=reasoning,
                    occasion=str(occasion),
                )
            )

        return RecommendResponse(recommendations=recs)


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------


def _parse_notes(*fields: object) -> list[str]:
    notes: list[str] = []
    for field in fields:
        if not field or str(field).lower() in ("nan", "none", ""):
            continue
        for note in str(field).split(","):
            note = note.strip().lower()
            if note:
                notes.append(note)
    return notes[:8]


def _str_or_none(val: object) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return None if s.lower() in ("nan", "none", "") else s
