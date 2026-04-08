from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
