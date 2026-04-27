from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from vibescents.perfume_scraper import search_perfumes  # noqa: E402

app = FastAPI(title="ScentAI Scraper API")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


class SearchRequest(BaseModel):
    perfumes: list[str] = Field(..., min_length=1, max_length=10)
    budget: float = Field(..., gt=0, le=10_000)


@app.post("/search")
def search(req: SearchRequest) -> list[dict]:
    try:
        print(f"[scraper] Searching {len(req.perfumes)} items, budget ${req.budget}")
        raw_results = search_perfumes(req.perfumes, req.budget)

        clean_results = []
        for i, res in enumerate(raw_results):
            if res is None:
                clean_results.append(
                    {
                        "name": req.perfumes[i],
                        "price": "N/A",
                        "store": "Unavailable",
                        "purchaseUrl": "#",
                        "thumbnail": None,
                    }
                )
            else:
                clean_results.append({**res, "name": req.perfumes[i]})

        print(f"[scraper] Returning {len(clean_results)} results")
        return clean_results

    except EnvironmentError:
        raise HTTPException(status_code=500, detail="Scraper configuration error")
    except Exception as e:
        print(f"[scraper] Unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error during scraping."
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("vibescents.scraper_app:app", host="127.0.0.1", port=8001)
