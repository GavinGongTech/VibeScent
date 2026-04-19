import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import clip_zero_shot


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 50)
    print("VIBESCENT WEB SERVER BOOTING UP", flush=True)
    print("=" * 50 + "\n", flush=True)
    clip_zero_shot.initialize_model()
    yield


app = FastAPI(title="ScentAI Model API", lifespan=lifespan)

_allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ContextInput(BaseModel):
    eventType: Optional[str] = None
    timeOfDay: Optional[str] = None
    mood: Optional[str] = None


class RecommendRequest(BaseModel):
    image: str
    mimeType: str
    context: ContextInput


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": clip_zero_shot.model is not None}


@app.post("/predict")
async def predict_fragrance(req: RecommendRequest):
    try:
        print("\n" + "*" * 40, flush=True)
        print("WEB: Request received. Processing image...", flush=True)

        raw = req.image
        image_data = base64.b64decode(raw.split(",")[1] if "," in raw else raw)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        context = req.context.model_dump(exclude_none=True)
        recommendations = clip_zero_shot.get_recommendations(img, context=context)

        print("WEB: Inference complete. Sending to Next.js.", flush=True)
        print("*" * 40 + "\n", flush=True)

        return {"recommendations": recommendations}

    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
