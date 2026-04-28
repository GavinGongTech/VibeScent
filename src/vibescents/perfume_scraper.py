import os
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPAPI_URL = "https://serpapi.com/search"

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = _REPO_ROOT / "artifacts" / "perfume_results.json"



def search_perfume(name: str, budget: float) -> list[dict]:
    if not SERPAPI_KEY:
        raise EnvironmentError("SERPAPI_KEY is not set")

    params = {
        "engine": "amazon",
        "k": f'"{name}" cologne OR "eau de parfum" OR "eau de toilette" OR perfume',
        "amazon_domain": "amazon.com",
        "api_key": SERPAPI_KEY,
        "num": "40",
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"[scraper] Request failed for '{name}': {exc}")
        return []

    results = []
    for item in data.get("organic_results", []):
        price = item.get("extracted_price")
        if price is None or price > budget:
            continue

        thumbnail_url = item.get("thumbnail", "")
        if thumbnail_url:
            thumbnail_url = re.sub(r'\._[^.]*_\.', '.', thumbnail_url)

        results.append(
            {
                "name": name,
                "title": item.get("title", ""),
                "price": float(price),
                "store": "Amazon",
                "url": item.get("link_clean", ""),
                "thumbnail": thumbnail_url,
                "asin": item.get("asin", ""),
                "rating": item.get("rating"),
                "reviews": item.get("reviews"),
                "prime": bool(item.get("prime", False)),
                "in_budget": True,
            }
        )

    # Filter out knockoff and inspired-by products
    knockoff_keywords = [
        "inspired by", "inspired-by", "type", "impression", 
        "alternative", "dupe", "our version", "similar to"
    ]
    results = [
        r for r in results 
        if not any(kw in r["title"].lower() for kw in knockoff_keywords)
    ]

    # Fuzzy relevance filter using SequenceMatcher
    def relevance_score(title: str, name: str) -> float:
        return SequenceMatcher(None, name.lower(), title.lower()).ratio()
    
    results = [
        r for r in results
        if relevance_score(r["title"], name) >= 0.3
    ]

    # Filter out non-perfume products
    non_perfume_keywords = [
        'deodorant', 'body wash', 'lotion', 'shower', 'shampoo',
        'conditioner', 'soap', 'candle', 'diffuser',
        'atomiser', 'atomizer', 'refill bottle', 'empty bottle',
        'body scrub', 'body mist', 'fragrance mist', 'body lotion',
        'hand cream', 'body cream', 'room spray', 'body butter',
        'hair', 'scrub', 'mist', 'spray', 'butter', 'wash',
        'moisturizer', 'balm', 'oil'
    ]
    results = [
        r for r in results
        if not any(kw in r['title'].lower() for kw in non_perfume_keywords)
    ]

    # Deprioritize samples/vials/minis by sorting them to bottom
    sample_keywords = ['sample', 'vial', 'mini', 'travel', 
                       '0.05', '1.5ml', '1ml', 'sampler']
    
    def is_sample(title):
        return any(k in title.lower() for k in sample_keywords)
    
    results.sort(key=lambda x: (
        is_sample(x['title']),
        -relevance_score(x['title'], name),
        x['price']
    ))
    return results[:3]


def search_perfumes(perfumes: list[str], budget: float) -> list[Optional[list[dict]]]:
    all_results: list[Optional[list[dict]]] = [None] * len(perfumes)

    with ThreadPoolExecutor(max_workers=min(len(perfumes), 10)) as executor:
        futures = {
            executor.submit(search_perfume, name, budget): idx
            for idx, name in enumerate(perfumes)
        }

        for future in futures:
            idx = futures[future]
            try:
                matches = future.result()
                all_results[idx] = matches if matches else None
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

