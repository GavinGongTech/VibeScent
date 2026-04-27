import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import requests

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPAPI_URL = "https://serpapi.com/search"

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = _REPO_ROOT / "artifacts" / "perfume_results.json"

PRIORITY_STORES = [
    "sephora",
    "nordstrom",
    "macy",
    "bloomingdale",
    "neiman marcus",
    "ulta",
]


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
    all_results: list[Optional[dict]] = [None] * len(perfumes)

    with ThreadPoolExecutor(max_workers=min(len(perfumes), 10)) as executor:
        futures = {
            executor.submit(search_perfume, name, budget): idx
            for idx, name in enumerate(perfumes)
        }

        for future in futures:
            idx = futures[future]
            try:
                matches = future.result()
                all_results[idx] = matches[0] if matches else None
            except Exception:
                all_results[idx] = None

    return all_results


if __name__ == "__main__":
    import sys

    perfumes = sys.argv[1:] or ["Baccarat Rouge 540", "Black Orchid"]
    budget = 150.0
    results = search_perfumes(perfumes, budget)
    for r in results:
        print(r)
