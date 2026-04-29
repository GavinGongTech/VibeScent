from __future__ import annotations

import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPAPI_URL = "https://serpapi.com/search"

PRIORITY_STORES = [
    "sephora",
    "ulta",
    "nordstrom",
    "macy's",
    "bloomingdale's",
    "saks fifth avenue",
    "neiman marcus",
    "fragrancex",
    "fragrancenet",
    "jomashop",
    "amazon",
]

KNOCKOFF_KEYWORDS = {
    "inspired by",
    "inspired-by",
    "impression",
    "dupe",
    "our version",
    "similar to",
    "compare to",
    "type",
}

NON_PERFUME_KEYWORDS = {
    "deodorant",
    "body wash",
    "lotion",
    "shower gel",
    "shampoo",
    "conditioner",
    "soap",
    "candle",
    "diffuser",
    "atomizer",
    "atomiser",
    "refill bottle",
    "empty bottle",
    "body scrub",
    "hand cream",
    "body cream",
    "room spray",
    "body butter",
    "moisturizer",
    "balm",
    "massage oil",
    "wallflower",
    "wallflowers",
    "refill",
    "plug-in",
    "plugin",
    "wax melt",
    "laundry",
    "detergent",
    "fabric softener",
    "incense",
}

SAMPLE_KEYWORDS = {
    "sample",
    "vial",
    "mini",
    "travel",
    "decant",
    "tester",
    "sampler",
    "1ml",
    "1.5ml",
    "2ml",
    "5ml",
}

OUTPUT_PATH = _REPO_ROOT / "artifacts" / "perfume_results.json"


def _parse_price(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    cleaned = re.sub(r"[^0-9.]", "", str(raw).replace(",", ""))
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def _store_priority(store: str | None) -> int:
    normalized = (store or "").strip().lower()
    for idx, preferred in enumerate(PRIORITY_STORES):
        if preferred in normalized:
            return idx
    return len(PRIORITY_STORES)


def _normalize_text(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value)
    ascii_value = ascii_value.encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^a-z0-9]+", " ", ascii_value.lower())
    return re.sub(r"\s+", " ", ascii_value).strip()


def _query_variants(name: str) -> list[str]:
    cleaned = " ".join(name.split())
    normalized = _normalize_text(cleaned)

    variants = [
        f'"{cleaned}" perfume fragrance',
        f"{cleaned} perfume fragrance",
    ]

    if normalized and normalized != cleaned.lower():
        variants.append(f'"{normalized}" perfume fragrance')

    if " - " in cleaned:
        variants.append(f'{cleaned.replace(" - ", " ")} perfume fragrance')

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.strip().lower()
        if key and key not in seen:
            deduped.append(variant)
            seen.add(key)
    return deduped


def _relevance_score(title: str, name: str) -> float:
    normalized_title = _normalize_text(title)
    normalized_name = _normalize_text(name)
    if not normalized_title or not normalized_name:
        return 0.0

    title_tokens = set(normalized_title.split())
    name_tokens = set(normalized_name.split())
    overlap = len(title_tokens & name_tokens) / max(len(name_tokens), 1)
    ratio = SequenceMatcher(None, normalized_name, normalized_title).ratio()
    return max(overlap, ratio)


def _contains_any(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _is_perfume_listing(title: str) -> bool:
    lowered = title.lower()
    if _contains_any(lowered, KNOCKOFF_KEYWORDS):
        return False
    if _contains_any(lowered, NON_PERFUME_KEYWORDS):
        return False
    return True


def _clean_thumbnail_url(item: dict) -> str:
    url = item.get("thumbnail") or ""
    if not url:
        thumbs = item.get("serpapi_thumbnails") or []
        if thumbs and isinstance(thumbs, list):
            url = str(thumbs[0] or "")
    if not url:
        return ""
    return re.sub(r"\._[^.]*_\.", ".", url)


def _normalize_result(item: dict, name: str, budget: float) -> dict | None:
    title = str(item.get("title", "")).strip()
    if not title:
        return None

    if not _is_perfume_listing(title):
        return None

    relevance = _relevance_score(title, name)
    if len(_normalize_text(title)) >= 6 and relevance < 0.55:
        return None

    price = (
        _parse_price(item.get("price"))
        or _parse_price(item.get("extracted_price"))
        or _parse_price(item.get("price_from"))
    )
    store = str(item.get("source") or item.get("seller") or item.get("store") or "").strip()
    url = str(item.get("product_link") or item.get("link") or item.get("offer_link") or "").strip()
    thumbnail = _clean_thumbnail_url(item)

    return {
        "name": name,
        "title": title,
        "price": price,
        "store": store or "Unknown retailer",
        "url": url,
        "purchaseUrl": url,
        "thumbnail": thumbnail,
        "in_budget": bool(price is not None and price <= budget),
        "inBudget": bool(price is not None and price <= budget),
        "relevance": relevance,
        "is_sample": _contains_any(title, SAMPLE_KEYWORDS),
    }


def _fetch_query(query: str) -> list[dict]:
    params = {
        "engine": "google_shopping",
        "q": query,
        "gl": "us",
        "hl": "en",
        "num": "20",
        "api_key": SERPAPI_KEY,
    }
    response = requests.get(SERPAPI_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return payload.get("shopping_results", []) or []


def search_perfume(name: str, budget: float) -> list[dict]:
    if not SERPAPI_KEY:
        raise EnvironmentError("SERPAPI_KEY is not set")

    raw_results: list[dict] = []
    try:
        for query in _query_variants(name):
            items = _fetch_query(query)
            raw_results.extend(items)
            normalized_hits = [
                result
                for result in (_normalize_result(item, name, budget) for item in items)
                if result is not None
            ]
            if any(result["in_budget"] for result in normalized_hits):
                break
    except requests.RequestException as exc:
        print(f"[scraper] Request failed for '{name}': {exc}")
        return []

    deduped: dict[tuple[str, str], dict] = {}
    for item in raw_results:
        normalized = _normalize_result(item, name, budget)
        if normalized is None:
            continue
        key = (
            normalized["url"].strip().lower(),
            normalized["title"].strip().lower(),
        )
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = normalized
            continue

        keep_new = (
            (normalized["thumbnail"] and not existing["thumbnail"])
            or (
                normalized["price"] is not None
                and existing["price"] is None
            )
            or normalized["relevance"] > existing["relevance"]
        )
        if keep_new:
            deduped[key] = normalized

    results = list(deduped.values())
    results.sort(
        key=lambda item: (
            not item["in_budget"],
            item["is_sample"],
            _store_priority(item["store"]),
            -item["relevance"],
            not bool(item["thumbnail"]),
            float(item["price"]) if item["price"] is not None else float("inf"),
        )
    )

    in_budget_results = [item for item in results if item["in_budget"]]
    if in_budget_results:
        results = in_budget_results

    trimmed: list[dict] = []
    for item in results[:3]:
        trimmed.append(
            {
                "name": item["name"],
                "title": item["title"],
                "price": item["price"],
                "store": item["store"],
                "url": item["url"],
                "purchaseUrl": item["purchaseUrl"],
                "thumbnail": item["thumbnail"] or "",
                "in_budget": item["in_budget"],
                "inBudget": item["inBudget"],
            }
        )
    return trimmed


def search_perfumes(perfumes: list[str], budget: float) -> list[Optional[list[dict]]]:
    if not perfumes:
        return []

    all_results: list[Optional[list[dict]]] = [None] * len(perfumes)

    with ThreadPoolExecutor(max_workers=min(len(perfumes), 10)) as executor:
        futures = {
            executor.submit(search_perfume, name, budget): idx
            for idx, name in enumerate(perfumes)
        }

        for future, idx in futures.items():
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
    for result in results:
        print(result)
