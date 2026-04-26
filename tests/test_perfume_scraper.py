from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vibescents.perfume_scraper import (
    PRIORITY_STORES,
    _parse_price,
    _store_priority,
    search_perfume,
    search_perfumes,
)


# --- pure helpers ---


def test_store_priority_sephora_is_first() -> None:
    assert _store_priority("Sephora") == 0


def test_store_priority_nordstrom_low_rank() -> None:
    rank = _store_priority("nordstrom rack")
    assert rank < len(PRIORITY_STORES)


def test_store_priority_unknown_returns_max() -> None:
    assert _store_priority("Some Random Boutique") == len(PRIORITY_STORES)


def test_parse_price_simple() -> None:
    assert _parse_price("9.99") == pytest.approx(89.99)


def test_parse_price_with_commas() -> None:
    assert _parse_price(",299.00") == pytest.approx(1299.0)


def test_parse_price_no_dollar_sign() -> None:
    assert _parse_price("45.50") == pytest.approx(45.5)


def test_parse_price_empty_string() -> None:
    assert _parse_price("") is None


def test_parse_price_no_digits() -> None:
    assert _parse_price("N/A") is None


# --- search_perfume ---


def test_search_perfume_no_api_key_raises() -> None:
    with patch("vibescents.perfume_scraper.SERPAPI_KEY", ""):
        with pytest.raises(EnvironmentError, match="SERPAPI_KEY"):
            search_perfume("Chanel No 5", 100.0)


def test_search_perfume_filters_over_budget() -> None:
    resp = MagicMock()
    resp.json.return_value = {
        "shopping_results": [
            {"price": "0.00", "source": "Sephora", "title": "A", "product_link": "http://a", "thumbnail": ""},
            {"price": "00.00", "source": "Nordstrom", "title": "B", "product_link": "http://b", "thumbnail": ""},
        ]
    }
    with patch("vibescents.perfume_scraper.SERPAPI_KEY", "fake-key"),          patch("requests.get", return_value=resp):
        results = search_perfume("Chanel No 5", 100.0)
    assert len(results) == 1
    assert results[0]["price"] == pytest.approx(50.0)


def test_search_perfume_sorts_by_priority_store() -> None:
    resp = MagicMock()
    resp.json.return_value = {
        "shopping_results": [
            {"price": "0.00", "source": "Some Store", "title": "A", "product_link": "http://a", "thumbnail": ""},
            {"price": "0.00", "source": "Sephora", "title": "B", "product_link": "http://b", "thumbnail": ""},
        ]
    }
    with patch("vibescents.perfume_scraper.SERPAPI_KEY", "fake-key"),          patch("requests.get", return_value=resp):
        results = search_perfume("Chanel No 5", 200.0)
    # Sephora should sort first despite higher price
    assert results[0]["store"] == "Sephora"


def test_search_perfume_request_error_returns_empty() -> None:
    import requests as req_mod
    with patch("vibescents.perfume_scraper.SERPAPI_KEY", "fake-key"),          patch("requests.get", side_effect=req_mod.RequestException("timeout")):
        results = search_perfume("Chanel No 5", 100.0)
    assert results == []


# --- search_perfumes ---


def test_search_perfumes_returns_one_per_fragrance(tmp_path) -> None:
    fake_result = {"name": "A", "price": 50.0, "store": "Sephora", "url": "http://a", "thumbnail": "", "in_budget": True}
    with patch("vibescents.perfume_scraper.search_perfume", return_value=[fake_result]),          patch("vibescents.perfume_scraper.OUTPUT_PATH", tmp_path / "out.json"):
        results = search_perfumes(["Fragrance A", "Fragrance B"], 100.0)
    assert len(results) == 2


def test_search_perfumes_none_for_no_match(tmp_path) -> None:
    with patch("vibescents.perfume_scraper.search_perfume", return_value=[]),          patch("vibescents.perfume_scraper.OUTPUT_PATH", tmp_path / "out.json"):
        results = search_perfumes(["Unknown"], 100.0)
    assert results == [None]


def test_search_perfumes_writes_json(tmp_path) -> None:
    fake_result = {"name": "A", "price": 50.0, "store": "Sephora"}
    out_path = tmp_path / "out.json"
    with patch("vibescents.perfume_scraper.search_perfume", return_value=[fake_result]),          patch("vibescents.perfume_scraper.OUTPUT_PATH", out_path):
        search_perfumes(["A"], 100.0)
    assert out_path.exists()
