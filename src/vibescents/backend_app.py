from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any, Protocol

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - optional local dependency
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

from vibescents.schemas import RecommendRequest, RecommendResponse


class RecommendationEngine(Protocol):
    def recommend(
        self, *, request: RecommendRequest, image_bytes: bytes
    ) -> RecommendResponse:
        """Generate recommendation response for the frontend contract."""


@dataclass
class UnconfiguredRecommendationEngine:
    message: str = (
        "Recommendation engine not configured. Inject a concrete engine with "
        "set_recommendation_engine() before calling /recommend."
    )

    def recommend(
        self, *, request: RecommendRequest, image_bytes: bytes
    ) -> RecommendResponse:
        raise RuntimeError(self.message)


def decode_request_image(payload: RecommendRequest) -> bytes:
    try:
        return base64.b64decode(payload.image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload in request.image.") from exc


def create_app(engine: RecommendationEngine | None = None) -> Any:
    if FastAPI is None:
        raise ImportError(
            "FastAPI is not installed. Install it in Colab with notebooks/requirements.colab.txt."
        )
    app = FastAPI(title="VibeScent backend", version="0.1.0")
    app.state.recommendation_engine = engine or UnconfiguredRecommendationEngine()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(payload: RecommendRequest) -> RecommendResponse:
        try:
            image_bytes = decode_request_image(payload)
            return app.state.recommendation_engine.recommend(
                request=payload,
                image_bytes=image_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def set_recommendation_engine(app: FastAPI, engine: RecommendationEngine) -> None:
    app.state.recommendation_engine = engine


app = create_app() if FastAPI is not None else None


def create_configured_app(settings: Any = None) -> Any:
    """Production entry point — injects VibeScoreEngine from disk artifacts.

    Launch with:
        uvicorn vibescents.backend_app:configured_app --host 0.0.0.0 --port 8000
    """
    from vibescents.engine import VibeScoreEngine
    from vibescents.settings import Settings

    s = settings or Settings.from_env()
    engine = VibeScoreEngine.from_artifacts(settings=s)
    engine._get_embedder()
    engine._get_clip()
    return create_app(engine=engine)


configured_app: Any = None  # populated at import time when VIBESCENT_AUTOSTART=1 is set
if __import__("os").getenv("VIBESCENT_AUTOSTART") == "1" and FastAPI is not None:
    configured_app = create_configured_app()
