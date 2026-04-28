from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from vibescents.bm25_scorer import BM25CorpusScorer
from vibescents.embeddings import Qwen3VLMultimodalEmbedder
from vibescents.fusion import fuse_scores, min_max_normalize, DEFAULT_FUSION_WEIGHTS
from vibescents.image_scoring import (
    CLIPImageScorer,
    GENDER_INDEX,
    FREQUENCY_INDEX,
    ImageHeadProbabilities,
)
from vibescents.io_utils import load_embeddings
from vibescents.query import context_to_query_string, build_candidate_text
from vibescents.schemas import (
    ContextInput,
    FragranceRecommendation,
    RecommendRequest,
    RecommendResponse,
    RetrievalCandidate,
)
from vibescents.settings import Settings
from vibescents.similarity import (
    cosine_similarity_matrix,
    normalize_rows,
    top_k_indices,
    mmr_select,
)
from vibescents.structured_scorer import compute_structured_scores

logger = logging.getLogger(__name__)

_SEASON_INDEX: dict[str, int] = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}


class VibeScoreEngine:
    """
    4-channel fusion + listwise LLM reranker recommendation engine — zero API keys required.

    Channels:
      text (0.30)       — Qwen3-VL-Embedding-8B text query vs corpus
      multimodal (0.25) — Qwen3-VL-Embedding-8B image+text query vs corpus
      image (0.30)      — CLIP ViT-L/14 zero-shot → vectorised NLL vs enriched-DF attributes
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
        self._clip: CLIPImageScorer | None = None
        self._embedder_lock = threading.Lock()
        self._clip_lock = threading.Lock()
        self._reranker: object = None
        self._reranker_lock = threading.Lock()
        self._bm25: BM25CorpusScorer | None = None
        self._formality_arr: np.ndarray = pd.to_numeric(
            corpus_df.get('formality', pd.Series(dtype=float)), errors='coerce'
        ).fillna(0.5).values
        self._fresh_warm_arr: np.ndarray = pd.to_numeric(
            corpus_df.get('fresh_warm', pd.Series(dtype=float)), errors='coerce'
        ).fillna(0.5).values

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

        _bm25: BM25CorpusScorer | None = None
        _text_col = next((c for c in ('retrieval_text', 'vibe_sentence', 'name') if c in df.columns), None)
        if _text_col is not None:
            try:
                _texts = df[_text_col].fillna('').astype(str).tolist()
                _bm25 = BM25CorpusScorer(_texts)
                if _bm25.available:
                    logger.info('BM25 index built from %s (%d docs)', _text_col, len(_texts))
            except Exception as _exc:
                logger.warning('BM25 index failed: %s', _exc)

        engine = cls(corpus_embeddings=embeddings, corpus_df=df, settings=s)
        engine._bm25 = _bm25
        return engine

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
        q_emb: np.ndarray | None = None
        if embedder is not None:
            try:
                q_emb = embedder.embed_multimodal_documents([query_str])
                q_emb = normalize_rows(q_emb.astype(np.float32))
                text_scores = cosine_similarity_matrix(q_emb, self._corpus_emb)[0]
            except Exception as exc:
                logger.warning("Text embedding failed, skipping text channel: %s", exc)

        _rerank_image = None
        if image_bytes:
            import os as _os
            import tempfile as _tmf
            try:
                with _tmf.NamedTemporaryFile(suffix=".jpg", delete=False) as _f:
                    _tmp = _f.name
                    _f.write(image_bytes)
                _rerank_image = _tmp
            except Exception as exc:
                logger.warning("Failed to save temporary image for reranking: %s", exc)

        # Image channel — CLIP zero-shot
        image_scores: np.ndarray | None = None
        try:
            head_probs = self._get_clip().score_image(image_bytes)
            image_scores = self._vectorised_image_scores(head_probs)
        except Exception as exc:
            logger.warning("CLIP scoring failed, skipping image channel: %s", exc)

        # Structured channel — arithmetic match, never fails
        structured_scores = compute_structured_scores(request.context, self._corpus_df)

        fused = self._fuse(text_scores, None, image_scores, structured_scores)

        # Apply hard filter — zero out excluded rows BEFORE top_k selection
        filter_mask = self._hard_filter(request.context)
        fused_filtered = fused.copy()
        fused_filtered[~filter_mask] = 0.0

        if self._bm25 is not None and self._bm25.available:
            try:
                bm25_norm = min_max_normalize(self._bm25.score(query_str))
                fused_filtered = 0.9 * fused_filtered + 0.1 * bm25_norm
            except Exception as _bm25_exc:
                logger.warning('BM25 scoring failed: %s', _bm25_exc)

        retrieval_k = min(20, len(self._corpus_df))
        top_k_arr = top_k_indices(fused_filtered, retrieval_k)

        rerank_results = None
        try:
            rerank_results = self._try_rerank(top_k_arr, query_str, _rerank_image)
            if rerank_results is None:
                if q_emb is not None:
                    top3_indices = mmr_select(q_emb[0], self._corpus_emb, top_k_arr, top_k=3)
                else:
                    top3_indices = top_k_arr[:3]
            else:
                top3_indices = np.array([int(r.fragrance_id) for r in rerank_results], dtype=int)
        finally:
            if _rerank_image is not None:
                try:
                    import os as _os
                    _os.unlink(_rerank_image)
                except OSError:
                    pass

        return self._build_response(top3_indices, fused, request.context, rerank_results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _EMBEDDER_UNAVAILABLE = object()
    _RERANKER_UNAVAILABLE = object()

    def _get_reranker(self):
        if self._reranker is self._RERANKER_UNAVAILABLE:
            return None
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    try:
                        from vibescents.reranker import Qwen3VLReranker
                        self._reranker = Qwen3VLReranker(self._settings)
                        logger.info('Reranker loaded')
                    except Exception as exc:
                        logger.warning('Reranker unavailable (%s), falling back to MMR', exc)
                        self._reranker = self._RERANKER_UNAVAILABLE
                        return None
        return self._reranker

    def _hard_filter(self, ctx: ContextInput) -> np.ndarray:
        mask = np.ones(len(self._corpus_df), dtype=bool)
        formality = self._formality_arr
        fresh_warm = self._fresh_warm_arr
        if ctx.eventType in ('Gala', 'Wedding'):
            mask &= formality >= 0.3
        elif ctx.eventType == 'Casual':
            mask &= formality <= 0.8
        if ctx.mood == 'Fresh':
            mask &= fresh_warm <= 0.7
        elif ctx.mood == 'Warm':
            mask &= fresh_warm >= 0.3
        if mask.sum() < 3:
            return np.ones(len(self._corpus_df), dtype=bool)
        return mask

    def _try_rerank(self, top_k_arr: np.ndarray, query_str: str, image_path: str | None) -> list | None:
        reranker = self._get_reranker()
        if reranker is None:
            return None
        try:
            candidates = [
                RetrievalCandidate(
                    fragrance_id=str(int(idx)),
                    retrieval_text=build_candidate_text(self._corpus_df.iloc[int(idx)]),
                )
                for idx in top_k_arr
            ]
            resp = reranker.rerank(occasion_text=query_str, candidates=candidates, image_path=image_path)
            return resp.results[:3]
        except Exception as exc:
            logger.warning('Rerank call failed: %s', exc)
            return None

    def _get_embedder(self) -> Qwen3VLMultimodalEmbedder | None:
        if self._embedder is self._EMBEDDER_UNAVAILABLE:
            return None
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    try:
                        from vibescents.embeddings import SentenceTransformerEmbedder
                        
                        _MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
                        self._embedder = SentenceTransformerEmbedder(_MINILM_MODEL)
                        logger.info(
                            "Using %s (CPU mode, multimodal channel disabled)",
                            _MINILM_MODEL,
                        )
                    except Exception as exc:
                        logger.warning("Embedder unavailable: %s", exc)
                        self._embedder = self._EMBEDDER_UNAVAILABLE
                        return None
        return self._embedder  # type: ignore[return-value]

    def _get_clip(self) -> CLIPImageScorer:
        if self._clip is None:
            with self._clip_lock:
                if self._clip is None:
                    self._clip = CLIPImageScorer()
        return self._clip

    def _vectorised_image_scores(self, probs: ImageHeadProbabilities) -> np.ndarray:
        """Score all corpus rows against CLIP probabilities using vectorised NLL."""
        df = self._corpus_df
        eps = 1e-8

        formality = pd.to_numeric(df["formality"], errors="coerce").fillna(0.5).values
        day_night = pd.to_numeric(df["day_night"], errors="coerce").fillna(0.5).values
        seasons = df["likely_season"].fillna("all-season").astype(str).values
        genders = (
            df["gender"].fillna("neutral").astype(str).values
            if "gender" in df.columns
            else None
        )
        frequencies = (
            df["frequency"].fillna("everyday").astype(str).values
            if "frequency" in df.columns
            else None
        )

        # Map each fragrance's attributes → discrete class indices
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

        # Gender: default to neutral (2) when column is absent or unparseable
        _default_gender = GENDER_INDEX["neutral"]
        gender_targets = (
            np.array(
                [GENDER_INDEX.get(g.strip().lower(), _default_gender) for g in genders],
                dtype=int,
            )
            if genders is not None
            else np.full(len(df), _default_gender, dtype=int)
        )

        # Frequency: default to everyday (1) when column is absent or unparseable
        _default_freq = FREQUENCY_INDEX["everyday"]
        frequency_targets = (
            np.array(
                [
                    FREQUENCY_INDEX.get(f.strip().lower(), _default_freq)
                    for f in frequencies
                ],
                dtype=int,
            )
            if frequencies is not None
            else np.full(len(df), _default_freq, dtype=int)
        )

        # Look up the CLIP probability assigned to each fragrance's target class
        formal_p = np.clip(probs.formal[formal_targets], eps, 1.0)
        time_p = np.clip(probs.time[time_targets], eps, 1.0)
        season_p = np.clip(probs.season[season_targets], eps, 1.0)
        gender_p = np.clip(probs.gender[gender_targets], eps, 1.0)
        frequency_p = np.clip(probs.frequency[frequency_targets], eps, 1.0)

        nll = -(
            np.log(formal_p)
            + np.log(season_p)
            + np.log(time_p)
            + np.log(gender_p)
            + np.log(frequency_p)
        )
        return np.exp(-nll).astype(np.float32)

    def _fuse(
        self,
        text: np.ndarray | None,
        multi: np.ndarray | None,
        image: np.ndarray | None,
        structured: np.ndarray,
    ) -> np.ndarray:
        # Build dynamic weights dict based on what signals are available,
        # scaling up available signals to fill any missing weight gaps.
        available = {"structured": structured}
        weights = {"structured": DEFAULT_FUSION_WEIGHTS["structured"]}
        
        if text is not None:
            available["text"] = text
            weights["text"] = DEFAULT_FUSION_WEIGHTS["text"]
            
        if image is not None:
            available["image"] = image
            weights["image"] = DEFAULT_FUSION_WEIGHTS["image"]
            
        # If multimodal were ever restored, it would fall back gracefully
        if multi is not None:
            available["multimodal"] = multi
            weights["multimodal"] = 0.25
            
        # Normalize weights to sum to 1.0
        total_w = sum(weights.values())
        norm_weights = {k: v / total_w for k, v in weights.items()}
        
        return fuse_scores(available, weights=norm_weights)

    def _build_response(
        self,
        top_indices: np.ndarray,
        fused_scores: np.ndarray,
        ctx: ContextInput,
        rerank_results: list | None = None,
    ) -> RecommendResponse:
        df = self._corpus_df
        recs: list[FragranceRecommendation] = []
        
        # Build map of reranker explanations if available
        explanations = {}
        if rerank_results is not None:
            for r in rerank_results:
                explanations[r.fragrance_id] = r.explanation

        for rank, idx in enumerate(top_indices, start=1):
            row = df.iloc[int(idx)]
            notes = _parse_notes(
                row.get("top_notes"), row.get("middle_notes"), row.get("base_notes")
            )
            occasion = (
                _str_or_none(row.get("likely_occasion")) or ctx.eventType or "Evening"
            )
            
            # Prefer reranker explanation, fallback to dataset vibe_sentence, fallback to default
            frag_id = str(row.get("fragrance_id", ""))
            reasoning = explanations.get(frag_id)
            if not reasoning or "continuous batching" in reasoning:
                reasoning = _str_or_none(row.get("vibe_sentence"))
            if not reasoning:
                reasoning = "A distinctive fragrance selected for your look."
                
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
