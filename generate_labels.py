"""Generate synthetic relevance labels for Learning-to-Rank weight training.

Strategy:
- Sample 50 diverse user contexts (eventType × timeOfDay × mood combinations)
- For each context, score 5 fragrances in a SINGLE API call (batched prompt)
- 2 of 5 are "good match" candidates (filtered by metadata), 3 are random
- 12-second delay between API calls → ~5 RPM (well under free-tier limit)
- 50 API calls total → ~10 minutes, producing ~250 (context, fragrance, score) pairs
- Output: labels.csv ready for learn_weights.py
"""
import json
import os
import random
import sys
import time
import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.path.insert(0, "src")

from vibescents.query import build_candidate_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env")

CORPUS_PATH = os.environ.get(
    "CORPUS_METADATA_PATH",
    "data/vibescent_enriched.csv",
)

# Safe rate-limit config for Nvidia free tier
REQUESTS_PER_BATCH = 5        # fragrances scored per single API call
DELAY_BETWEEN_CALLS = 12      # seconds between calls → ~5 RPM
NUM_CONTEXTS = 50             # total diverse contexts to evaluate
FRAGRANCES_PER_CONTEXT = 5   # 2 good-match + 3 random per context
OUTPUT_PATH = "data/ltr_labels.csv"

# ── Diverse context grid ──────────────────────────────────────────────────────
EVENT_TYPES = ["Gala", "Date Night", "Casual", "Business", "Wedding", "Festival"]
TIMES = ["Morning", "Afternoon", "Evening", "Night"]
MOODS = ["Bold", "Subtle", "Fresh", "Warm", "Mysterious"]

# ── Nvidia client ─────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
)

# ── Load corpus ───────────────────────────────────────────────────────────────
log.info("Loading corpus from %s", CORPUS_PATH)
df = pd.read_csv(CORPUS_PATH, low_memory=False)
df = df.dropna(subset=["name"]).reset_index(drop=True)
log.info("Loaded %d fragrances", len(df))

# ── Helpers ───────────────────────────────────────────────────────────────────
def sample_contexts(n: int) -> list[dict]:
    """Generate n diverse (eventType, timeOfDay, mood) dicts."""
    contexts = []
    for event in EVENT_TYPES:
        for time_of_day in TIMES:
            for mood in MOODS:
                contexts.append({"eventType": event, "timeOfDay": time_of_day, "mood": mood})
    random.shuffle(contexts)
    return contexts[:n]


def select_fragrances(context: dict, df: pd.DataFrame, n_good: int = 2, n_random: int = 3) -> pd.DataFrame:
    """Pick n_good metadata-matching fragrances + n_random random ones."""
    event = context["eventType"]
    time_of_day = context["timeOfDay"]
    mood = context["mood"]

    # Simple metadata heuristics for "good" candidates
    mask = pd.Series([True] * len(df))
    if event in ("Gala", "Wedding", "Business"):
        if "formality" in df.columns:
            mask &= df["formality"].fillna(0.5) > 0.6
    elif event in ("Casual", "Festival"):
        if "formality" in df.columns:
            mask &= df["formality"].fillna(0.5) < 0.4

    if time_of_day in ("Evening", "Night"):
        if "day_night" in df.columns:
            mask &= df["day_night"].fillna(0.5) > 0.5

    if mood == "Fresh":
        if "fresh_warm" in df.columns:
            mask &= df["fresh_warm"].fillna(0.5) < 0.4
    elif mood == "Warm":
        if "fresh_warm" in df.columns:
            mask &= df["fresh_warm"].fillna(0.5) > 0.6

    good_pool = df[mask]
    if len(good_pool) < n_good:
        good_pool = df

    good = good_pool.sample(min(n_good, len(good_pool)), replace=False)
    random_pool = df[~df.index.isin(good.index)]
    random_sample = random_pool.sample(min(n_random, len(random_pool)), replace=False)

    return pd.concat([good, random_sample]).sample(frac=1)  # shuffle order


def build_batch_prompt(context: dict, fragrances: pd.DataFrame) -> str:
    """Ask Gemma to score all fragrances in one shot as JSON."""
    occasion_text = (
        f"{context['eventType']} | {context['timeOfDay']} | mood: {context['mood']}"
    )
    frag_list = []
    for i, (_, row) in enumerate(fragrances.iterrows(), 1):
        text = build_candidate_text(row)
        frag_list.append(f"[{i}] {text}")
    fragrances_block = "\n".join(frag_list)

    return f"""You are an expert fragrance consultant. Rate how well each fragrance matches the given occasion.

Occasion: {occasion_text}

Fragrances to score:
{fragrances_block}

For each fragrance [1] through [{len(frag_list)}], give a relevance score from 0.0 to 1.0:
- 1.0 = perfect match for this occasion and mood
- 0.5 = neutral / could work
- 0.0 = completely wrong for this occasion

Respond ONLY with valid JSON in this exact format, no other text:
{{"scores": [0.0, 0.0, 0.0, 0.0, 0.0]}}

The array must have exactly {len(frag_list)} numbers."""


def call_api_with_backoff(prompt: str, attempt: int = 0) -> list[float] | None:
    """Call Gemma, parse JSON scores. Retry once on failure."""
    try:
        resp = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        scores = data["scores"]
        return [float(s) for s in scores]
    except Exception as e:
        if attempt < 1:
            log.warning("API call failed (%s), retrying after 30s...", e)
            time.sleep(30)
            return call_api_with_backoff(prompt, attempt + 1)
        log.error("API call failed after retry: %s", e)
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────
contexts = sample_contexts(NUM_CONTEXTS)
log.info("Generated %d contexts. Starting scoring (~%d min)...",
         len(contexts), len(contexts) * DELAY_BETWEEN_CALLS // 60)

records = []
for i, context in enumerate(contexts, 1):
    fragrances = select_fragrances(context, df, n_good=2, n_random=3)
    prompt = build_batch_prompt(context, fragrances)

    log.info("[%d/%d] Scoring for: %s | %s | %s",
             i, len(contexts), context["eventType"], context["timeOfDay"], context["mood"])

    scores = call_api_with_backoff(prompt)

    if scores and len(scores) == len(fragrances):
        for j, (_, row) in enumerate(fragrances.iterrows()):
            records.append({
                "event_type": context["eventType"],
                "time_of_day": context["timeOfDay"],
                "mood": context["mood"],
                "fragrance_id": row.get("fragrance_id", str(row.name)),
                "name": row.get("name", ""),
                "brand": row.get("brand", ""),
                "retrieval_text": build_candidate_text(row),
                "relevance_score": scores[j],
            })
        log.info("  Scores: %s", [round(s, 2) for s in scores])
    else:
        log.warning("  Skipping — invalid response")

    # Rate limit: 12s between requests = ~5 RPM (free tier safe)
    if i < len(contexts):
        time.sleep(DELAY_BETWEEN_CALLS)

# ── Save ──────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(records)
os.makedirs("data", exist_ok=True)
out_df.to_csv(OUTPUT_PATH, index=False)
log.info("Saved %d labelled pairs to %s", len(out_df), OUTPUT_PATH)
log.info("Now run: uv run learn_weights.py")
