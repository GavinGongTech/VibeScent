from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")

from backend.perfume_scraper import search_perfumes  # noqa: E402 — must load env first

app = FastAPI(title="ScentAI Scraper API")


class SearchRequest(BaseModel):
    perfumes: list[str]
    budget: float


@app.post("/search")
def search(req: SearchRequest) -> list[dict]:
    if not req.perfumes:
        raise HTTPException(status_code=400, detail="perfumes list must not be empty")
    if req.budget <= 0:
        raise HTTPException(status_code=400, detail="budget must be greater than 0")
    try:
        return search_perfumes(req.perfumes, req.budget)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
