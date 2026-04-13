from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FragranceRecord(BaseModel):
    fragrance_id: str
    brand: str | None = None
    name: str | None = None
    retrieval_text: str
    display_text: str | None = None
    accords: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    season_tags: list[str] = Field(default_factory=list)
    occasion_tags: list[str] = Field(default_factory=list)
    formality_score: float | None = None
    fresh_warm_score: float | None = None
    day_night_score: float | None = None


class RetrievalCandidate(BaseModel):
    fragrance_id: str
    name: str | None = None
    brand: str | None = None
    retrieval_text: str
    display_text: str | None = None
    baseline_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankResult(BaseModel):
    fragrance_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    formality_score: float = Field(ge=0.0, le=1.0)
    season_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    explanation: str


class RerankResponse(BaseModel):
    results: list[RerankResult]


class BenchmarkCaseLabel(BaseModel):
    case_id: str
    occasion_text: str
    target_formality: str
    target_season: str
    target_day_night: str
    target_fresh_warm: str
    acceptable_accords: list[str] = Field(default_factory=list)
    acceptable_note_families: list[str] = Field(default_factory=list)
    disallowed_traits: list[str] = Field(default_factory=list)
    example_good_fragrances: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class BenchmarkCaseDraft(BaseModel):
    case_id: str
    brief: str
    labels: BenchmarkCaseLabel


class BenchmarkGenerationResponse(BaseModel):
    labels: list[BenchmarkCaseLabel]


SeasonLabel = Literal["spring", "summer", "fall", "winter", "all-season"]


class EnrichmentSchemaV2(BaseModel):
    likely_season: SeasonLabel
    likely_occasion: str
    formality: float = Field(ge=0.0, le=1.0)
    fresh_warm: float = Field(ge=0.0, le=1.0)
    day_night: float = Field(ge=0.0, le=1.0)
    character_tags: list[str] = Field(default_factory=list, min_length=3, max_length=5)
    vibe_sentence: str
    longevity: str
    projection: str
    mood_tags: list[str] = Field(default_factory=list, min_length=1)
    color_palette: list[str] = Field(default_factory=list, min_length=1)

    @field_validator("likely_occasion", "vibe_sentence", "longevity", "projection")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be empty.")
        return stripped

    @field_validator("character_tags", "mood_tags", "color_palette")
    @classmethod
    def _normalize_tag_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("List must contain at least one non-empty item.")
        return cleaned


class ContextInput(BaseModel):
    eventType: str | None = None
    timeOfDay: str | None = None
    mood: str | None = None
    customNotes: str | None = None


class RecommendRequest(BaseModel):
    image: str
    mimeType: Literal["image/jpeg", "image/png", "image/webp"]
    context: ContextInput


class FragranceRecommendation(BaseModel):
    rank: int = Field(ge=1)
    name: str
    house: str
    score: float = Field(ge=0.0)
    notes: list[str] = Field(default_factory=list)
    reasoning: str
    occasion: str


class RecommendResponse(BaseModel):
    recommendations: list[FragranceRecommendation] = Field(default_factory=list)
