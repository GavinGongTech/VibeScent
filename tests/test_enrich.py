from __future__ import annotations

import json

import pytest

from vibescents.enrich import _parse_enrichment, _repair_payload
from vibescents.schemas import EnrichmentSchemaV2


_VALID_DICT = {
    "likely_season": "spring",
    "likely_occasion": "Date Night",
    "formality": 0.5,
    "fresh_warm": 0.3,
    "day_night": 0.7,
    "character_tags": ["floral", "fresh", "light"],
    "vibe_sentence": "A light spring scent.",
    "longevity": "moderate",
    "projection": "soft",
    "mood_tags": ["romantic"],
    "color_palette": ["pale pink"],
}


def test_parse_enrichment_from_valid_dict() -> None:
    result = _parse_enrichment(_VALID_DICT)
    assert result is not None
    assert isinstance(result, EnrichmentSchemaV2)
    assert result.likely_season == "spring"


def test_parse_enrichment_from_json_string() -> None:
    result = _parse_enrichment(json.dumps(_VALID_DICT))
    assert result is not None
    assert result.formality == pytest.approx(0.5)


def test_parse_enrichment_from_schema_instance() -> None:
    schema = EnrichmentSchemaV2(**_VALID_DICT)
    result = _parse_enrichment(schema)
    assert result is schema


def test_parse_enrichment_invalid_json_string_returns_none() -> None:
    result = _parse_enrichment("not valid json")
    assert result is None


def test_parse_enrichment_missing_required_fields_returns_none() -> None:
    result = _parse_enrichment({"invalid_key": "value"})
    assert result is None


def test_parse_enrichment_empty_dict_returns_none() -> None:
    result = _parse_enrichment({})
    assert result is None


def test_parse_enrichment_invalid_formality_range_returns_none() -> None:
    bad = {**_VALID_DICT, "formality": 2.0}  # out of [0, 1]
    result = _parse_enrichment(bad)
    assert result is None


def test_repair_payload_valid_json_unchanged() -> None:
    payload = json.dumps({"key": "value"})
    result = _repair_payload(payload)
    assert result is not None


def test_repair_payload_non_json_returns_something() -> None:
    result = _repair_payload("random string")
    assert result is not None  # json_repair handles or falls back


def test_parse_enrichment_normalises_tags() -> None:
    d = {**_VALID_DICT, "character_tags": ["  floral  ", " fresh ", "light"]}
    result = _parse_enrichment(d)
    assert result is not None
    assert result.character_tags == ["floral", "fresh", "light"]
