from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibescents.schemas import (
    BenchmarkCaseLabel,
    ContextInput,
    EnrichmentSchemaV2,
    FragranceRecommendation,
    FragranceRecord,
    RecommendRequest,
    RecommendResponse,
    RerankResult,
)


def test_fragrance_record_defaults() -> None:
    """FragranceRecord field defaults (brand/accords/formality_score are None/empty by default)"""
    record = FragranceRecord(fragrance_id="f1", retrieval_text="text")
    assert record.brand is None
    assert record.accords == []
    assert record.formality_score is None


def test_rerank_result_validation() -> None:
    """RerankResult rejects overall_score > 1.0 and accepts boundary values 0.0 and 1.0"""
    # rejects overall_score > 1.0
    with pytest.raises(ValidationError):
        RerankResult(
            fragrance_id="f1",
            overall_score=1.1,
            formality_score=0.5,
            season_score=0.5,
            freshness_score=0.5,
            explanation="too high",
        )

    # accepts boundary values 0.0 and 1.0
    r0 = RerankResult(
        fragrance_id="f1",
        overall_score=0.0,
        formality_score=0.0,
        season_score=0.0,
        freshness_score=0.0,
        explanation="low",
    )
    assert r0.overall_score == 0.0

    r1 = RerankResult(
        fragrance_id="f1",
        overall_score=1.0,
        formality_score=1.0,
        season_score=1.0,
        freshness_score=1.0,
        explanation="high",
    )
    assert r1.overall_score == 1.0


def test_enrichment_schema_v2_construction() -> None:
    """EnrichmentSchemaV2 valid construction with all required fields"""
    enrichment = EnrichmentSchemaV2(
        likely_season="summer",
        likely_occasion="beach party",
        formality=0.2,
        fresh_warm=0.8,
        day_night=0.9,
        gender="neutral",
        frequency="everyday",
        character_tags=["citrus", "oceanic", "breezy"],
        vibe_sentence="A refreshing summer day at the coast.",
        longevity="moderate",
        projection="intimate",
        mood_tags=["energetic"],
        color_palette=["blue", "yellow"],
    )
    assert enrichment.likely_season == "summer"
    assert enrichment.character_tags == ["citrus", "oceanic", "breezy"]


def test_enrichment_schema_v2_validation() -> None:
    """EnrichmentSchemaV2 validation for empty strings and whitespace-only tags"""
    # vibe_sentence='   ' (whitespace only) raises ValidationError
    with pytest.raises(ValidationError):
        EnrichmentSchemaV2(
            likely_season="summer",
            likely_occasion="beach",
            formality=0.5,
            fresh_warm=0.5,
            day_night=0.5,
            character_tags=["tag1", "tag2", "tag3"],
            vibe_sentence="   ",
            longevity="long",
            projection="strong",
            mood_tags=["mood"],
            color_palette=["color"],
        )

    # mood_tags=['   '] (all whitespace) raises ValidationError
    with pytest.raises(ValidationError):
        EnrichmentSchemaV2(
            likely_season="summer",
            likely_occasion="beach",
            formality=0.5,
            fresh_warm=0.5,
            day_night=0.5,
            character_tags=["tag1", "tag2", "tag3"],
            vibe_sentence="vibe",
            longevity="long",
            projection="strong",
            mood_tags=["   "],
            color_palette=["color"],
        )


def test_enrichment_schema_v2_stripping() -> None:
    """EnrichmentSchemaV2 strips whitespace from character_tags"""
    enrichment = EnrichmentSchemaV2(
        likely_season="summer",
        likely_occasion="beach",
        formality=0.5,
        fresh_warm=0.5,
        day_night=0.5,
        character_tags=["  oud  ", " rose ", "musk"],
        vibe_sentence="vibe",
        longevity="long",
        projection="strong",
        mood_tags=["mood"],
        color_palette=["color"],
    )
    assert enrichment.character_tags == ["oud", "rose", "musk"]


def test_context_input_defaults() -> None:
    """ContextInput all fields default to None"""
    ctx = ContextInput()
    assert ctx.eventType is None
    assert ctx.timeOfDay is None
    assert ctx.mood is None
    assert ctx.customNotes is None


def test_recommend_request_validation() -> None:
    """RecommendRequest rejects mimeType='image/gif'"""
    with pytest.raises(ValidationError):
        RecommendRequest(
            image="base64str",
            mimeType="image/gif",  # type: ignore
            context=ContextInput(),
        )


def test_fragrance_recommendation_validation() -> None:
    """FragranceRecommendation rejects rank=0"""
    with pytest.raises(ValidationError):
        FragranceRecommendation(
            rank=0,
            name="Test",
            house="House",
            score=0.5,
            reasoning="reason",
            occasion="occasion",
        )


def test_recommend_response_defaults() -> None:
    """RecommendResponse defaults to empty recommendations list"""
    resp = RecommendResponse()
    assert resp.recommendations == []


def test_benchmark_case_label_validation() -> None:
    """BenchmarkCaseLabel rejects confidence=1.5"""
    with pytest.raises(ValidationError):
        BenchmarkCaseLabel(
            case_id="c1",
            occasion_text="event",
            target_formality="formal",
            target_season="winter",
            target_day_night="night",
            target_fresh_warm="warm",
            confidence=1.5,
        )
