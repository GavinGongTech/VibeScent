"""
Demo server — serves pre-cached locked responses, no GPU required.
Run:  uv run python -m serve_demo
"""
from __future__ import annotations

import json
from pathlib import Path

import uvicorn

from vibescents.backend_app import create_app
from vibescents.schemas import FragranceRecommendation, RecommendRequest, RecommendResponse

LOCKED = Path(__file__).parent / "artifacts" / "week4" / "locked_responses.json"


class StaticEngine:
    """Returns locked responses for every request — no model loading."""

    def __init__(self) -> None:
        data = json.loads(LOCKED.read_text())
        self._recs = [
            FragranceRecommendation(**r)
            for r in data["default"]["recommendations"]
        ]

    def recommend(self, *, request: RecommendRequest, image_bytes: bytes) -> RecommendResponse:
        return RecommendResponse(recommendations=self._recs)


app = create_app(engine=StaticEngine())

if __name__ == "__main__":
    print("\n  VibeScent demo server — static mode")
    print("  GET  http://localhost:8001/healthz")
    print("  POST http://localhost:8001/recommend\n")
    uvicorn.run("serve_demo:app", host="0.0.0.0", port=8001, reload=False)
