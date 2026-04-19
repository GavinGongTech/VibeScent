from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")

from perfume_scraper import search_perfumes  # noqa: E402 — must load env first

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
        # ---------------------------------------------------------
        # 🖨️ PRINT 1: INCOMING SEARCH REQUEST
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print(f"🛒 SCRAPER INITIATED: Searching for {len(req.perfumes)} items")
        print("="*50)
        print(f"Budget Cap: ${req.budget}")
        print(f"Targets: {', '.join(req.perfumes)}")
        print("Scraping Google Shopping... 🕵️‍♂️")

        # 1. Get the raw results from the scraper
        raw_results = search_perfumes(req.perfumes, req.budget)
        
        # 2. Sanitize the results so FastAPI and Next.js don't crash
        clean_results = []
        for res in raw_results:
            if res is None:
                # Provide a safe fallback dictionary if Google Shopping finds nothing
                clean_results.append({
                    "price": "N/A",
                    "store": "Unavailable",
                    "purchaseUrl": "#",
                    "thumbnail": "https://via.placeholder.com/150?text=Not+Found"
                })
            else:
                clean_results.append(res)

        # ---------------------------------------------------------
        # 🖨️ PRINT 2: OUTGOING RESULTS
        # ---------------------------------------------------------
        print("\n" + "-"*50)
        print("✅ SCRAPER RESULTS:")
        for i, res in enumerate(clean_results):
            name = req.perfumes[i]
            # Format the price nicely for the terminal
            price = f"${res['price']}" if isinstance(res['price'], (int, float)) else res['price']
            print(f"  {i+1}. {name}")
            print(f"     Found at: {res['store']} | Price: {price}")
        print("="*50 + "\n")

        return clean_results

    except EnvironmentError as exc:
        print(f"Scraper Environment Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as e:
        print(f"Unexpected Scraper Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during scraping.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
