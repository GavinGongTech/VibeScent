import json
import os
import re
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPAPI_URL = "https://serpapi.com/search"

OUTPUT_PATH = Path(__file__).parent / "outputs" / "perfume_results.json"

PRIORITY_STORES = ["sephora", "nordstrom", "macy", "bloomingdale", "neiman marcus", "ulta"]


def _store_priority(store: str) -> int:
    lower = store.lower()
    for i, name in enumerate(PRIORITY_STORES):
        if name in lower:
            return i
    return len(PRIORITY_STORES)


def _parse_price(price_str: str) -> Optional[float]:
    if not price_str:
        return None
    match = re.search(r"[\d,]+\.?\d*", price_str.replace(",", ""))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def search_perfume(name: str, budget: float) -> list[dict]:
    if not SERPAPI_KEY:
        raise EnvironmentError("SERPAPI_KEY is not set")

    params = {
        "engine": "google_shopping",
        "q": f"{name} perfume fragrance",
        "api_key": SERPAPI_KEY,
        "num": 20,
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"[scraper] Request failed for '{name}': {exc}")
        return []

    results = []
    for item in data.get("shopping_results", []):
        price = _parse_price(item.get("price", ""))
        if price is None or price > budget:
            continue
        results.append(
            {
                "name": name,
                "title": item.get("title", ""),
                "price": price,
                "store": item.get("source", ""),
                "url": item.get("product_link", ""),
                "thumbnail": item.get("thumbnail", ""),
                "in_budget": True,
            }
        )

    results.sort(key=lambda x: (_store_priority(x["store"]), x["price"]))
    return results


def search_perfumes(perfumes: list[str], budget: float) -> list[Optional[dict]]:
    all_results: list[Optional[dict]] = []

    for name in perfumes:
        matches = search_perfume(name, budget)
        all_results.append(matches[0] if matches else None)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"[scraper] Results saved to {OUTPUT_PATH}")

    return all_results


if __name__ == "__main__":
    import sys

    perfumes = sys.argv[1:] or ["Baccarat Rouge 540", "Black Orchid"]
    budget = 150.0
    results = search_perfumes(perfumes, budget)
    for r in results:
        print(r)
