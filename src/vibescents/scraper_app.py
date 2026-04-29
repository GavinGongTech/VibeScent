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
            if not res:
                clean_results.append(
                    {
                        "name": req.perfumes[i],
                        "price": "N/A",
                        "store": "Unavailable",
                        "url": "#",
                        "purchaseUrl": "#",
                        "thumbnail": None,
                        "in_budget": False,
                        "inBudget": False,
                    }
                )
            else:
                # Accept either the normal list[dict] payload or a single dict from tests/mocks.
                if isinstance(res, dict):
                    best = res
                else:
                    best = next((item for item in res if item), None) or {}
                clean_results.append({
                    "name": req.perfumes[i],
                    "price": best.get("price", "N/A"),
                    "store": best.get("store", "Unknown Retailer"),
                    "url": best.get("url", "#"),
                    "purchaseUrl": best.get("purchaseUrl", best.get("url", "#")),
                    "thumbnail": best.get("thumbnail") or None,
                    "in_budget": best.get("in_budget", best.get("inBudget", False)),
                    "inBudget": best.get("inBudget", best.get("in_budget", False)),
                    "rating": best.get("rating"),
                    "title": best.get("title", ""),
                })

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
