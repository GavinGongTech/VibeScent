from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from vibescents.scraper_app import app

client = TestClient(app)

_FAKE_RESULT = {
    "name": "Baccarat",
    "price": 80.0,
    "store": "Sephora",
    "url": "http://a",
    "thumbnail": "",
    "in_budget": True,
}


def test_search_returns_200_for_valid_request() -> None:
    with patch("vibescents.scraper_app.search_perfumes", return_value=[_FAKE_RESULT]):
        resp = client.post(
            "/search", json={"perfumes": ["Baccarat Rouge 540"], "budget": 150.0}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "Baccarat Rouge 540"


def test_search_empty_perfumes_returns_400() -> None:
    resp = client.post("/search", json={"perfumes": [], "budget": 100.0})
    assert resp.status_code == 400
    assert "perfumes" in resp.json()["detail"]


def test_search_zero_budget_returns_400() -> None:
    resp = client.post("/search", json={"perfumes": ["Fragrance A"], "budget": 0.0})
    assert resp.status_code == 400


def test_search_negative_budget_returns_400() -> None:
    resp = client.post("/search", json={"perfumes": ["Fragrance A"], "budget": -10.0})
    assert resp.status_code == 400


def test_search_serpapi_key_missing_returns_500() -> None:
    with patch(
        "vibescents.scraper_app.search_perfumes",
        side_effect=EnvironmentError("SERPAPI_KEY is not set"),
    ):
        resp = client.post(
            "/search", json={"perfumes": ["Fragrance A"], "budget": 100.0}
        )
    assert resp.status_code == 500
    assert "SERPAPI_KEY" in resp.json()["detail"]


def test_search_unexpected_exception_returns_500() -> None:
    with patch(
        "vibescents.scraper_app.search_perfumes", side_effect=RuntimeError("boom")
    ):
        resp = client.post(
            "/search", json={"perfumes": ["Fragrance A"], "budget": 100.0}
        )
    assert resp.status_code == 500


def test_search_none_result_uses_fallback() -> None:
    with patch("vibescents.scraper_app.search_perfumes", return_value=[None]):
        resp = client.post(
            "/search", json={"perfumes": ["Unknown Scent"], "budget": 100.0}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["store"] == "Unavailable"
    assert data[0]["price"] == "N/A"
    assert data[0]["name"] == "Unknown Scent"


def test_search_multiple_fragrances() -> None:
    results = [_FAKE_RESULT.copy(), None]
    with patch("vibescents.scraper_app.search_perfumes", return_value=results):
        resp = client.post("/search", json={"perfumes": ["A", "B"], "budget": 200.0})
    assert resp.status_code == 200
    assert len(resp.json()) == 2
