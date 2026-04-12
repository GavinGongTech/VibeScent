"""Enrich fragrance dataset with LLM-generated vibe attributes.

Uses gemini-3-flash-preview to add season, occasion, formality,
fresh/warm, day/night, character tags, and a vibe sentence to each
fragrance based on its notes, accords, and other metadata.

Outputs data/vibescent_enriched.csv with a `retrieval_text` column
ready for embedding.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vibescents.settings import Settings

ENRICHMENT_MODEL = "gemini-3-flash-preview"
BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 1.0  # seconds


ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "likely_season": {
            "type": "string",
            "enum": ["spring", "summer", "fall", "winter", "all-season"],
        },
        "likely_occasion": {
            "type": "string",
        },
        "formality": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "fresh_warm": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "day_night": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "character_tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "vibe_sentence": {
            "type": "string",
        },
    },
    "required": [
        "likely_season",
        "likely_occasion",
        "formality",
        "fresh_warm",
        "day_night",
        "character_tags",
        "vibe_sentence",
    ],
}

SYSTEM_PROMPT = """\
You are a fragrance expert. Given a perfume's metadata, generate structured \
vibe attributes that describe its character, mood, and ideal context.

Rules:
- formality: 0.0 = very casual, 1.0 = black tie formal
- fresh_warm: 0.0 = crisp and fresh, 1.0 = warm and cozy
- day_night: 0.0 = bright daytime, 1.0 = evening/night
- character_tags: 3-5 adjectives capturing the fragrance's personality
- vibe_sentence: one sentence describing the mood, outfit style, and setting \
this fragrance pairs best with
- likely_occasion: one primary occasion (e.g. "casual outing", "date night", \
"formal event", "office", "beach day", "evening party")
- likely_season: the single best season, or "all-season"

Base your answers on the fragrance notes, accords, and category. \
If data is sparse, make reasonable inferences from whatever is available.
"""


def _build_prompt(row: pd.Series) -> str:
    parts = [f"Name: {row['name']}"]
    if pd.notna(row.get("brand")):
        parts.append(f"Brand: {row['brand']}")
    if pd.notna(row.get("top_notes")):
        parts.append(f"Top notes: {row['top_notes']}")
    if pd.notna(row.get("middle_notes")):
        parts.append(f"Middle notes: {row['middle_notes']}")
    if pd.notna(row.get("base_notes")):
        parts.append(f"Base notes: {row['base_notes']}")
    if pd.notna(row.get("main_accords")):
        parts.append(f"Accords: {row['main_accords']}")
    if pd.notna(row.get("gender")):
        parts.append(f"Gender: {row['gender']}")
    if pd.notna(row.get("concentration")):
        parts.append(f"Concentration: {row['concentration']}")
    if pd.notna(row.get("category")):
        parts.append(f"Category: {row['category']}")
    return "\n".join(parts)


def _parse_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _build_retrieval_text(row: pd.Series) -> str:
    parts = []

    # Identity
    name = row.get("name", "")
    brand = row.get("brand", "")
    if pd.notna(brand) and brand:
        parts.append(f"Brand: {brand} | Name: {name}")
    else:
        parts.append(f"Name: {name}")

    # Accords
    if pd.notna(row.get("main_accords")):
        parts.append(f"Accords: {row['main_accords']}")

    # Notes
    note_parts = []
    for col, label in [("top_notes", "Top"), ("middle_notes", "Heart"), ("base_notes", "Base")]:
        if pd.notna(row.get(col)):
            note_parts.append(f"{label}: {row[col]}")
    if note_parts:
        parts.append(" | ".join(note_parts))

    # Enriched fields
    if pd.notna(row.get("likely_season")):
        parts.append(f"Season: {row['likely_season']}")
    if pd.notna(row.get("likely_occasion")):
        parts.append(f"Best for: {row['likely_occasion']}")
    if pd.notna(row.get("formality")):
        level = "low" if row["formality"] < 0.33 else "medium" if row["formality"] < 0.67 else "high"
        parts.append(f"Formality: {level}")
    tags = row.get("character_tags")
    if tags is not None and not (isinstance(tags, float) and np.isnan(tags)):
        if isinstance(tags, str) and tags:
            parts.append(f"Character: {tags}")
        elif isinstance(tags, list) and tags:
            parts.append(f"Character: {', '.join(tags)}")
    if pd.notna(row.get("vibe_sentence")):
        parts.append(f"Vibe: {row['vibe_sentence']}")

    return " | ".join(parts)


def enrich_dataframe(
    df: pd.DataFrame,
    *,
    max_rows: int | None = None,
    resume_from: int = 0,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """Enrich a fragrance dataframe with LLM-generated attributes."""
    settings = Settings.from_env()
    if not settings.api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.api_key)

    work = df.copy()
    end = min(len(work), resume_from + max_rows) if max_rows else len(work)
    subset = work.iloc[resume_from:end]

    enriched_cols = [
        "likely_season", "likely_occasion", "formality",
        "fresh_warm", "day_night", "character_tags", "vibe_sentence",
    ]
    for col in enriched_cols:
        if col not in work.columns:
            work[col] = pd.Series([None] * len(work), dtype="object")

    total = len(subset)
    processed = 0
    failed = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = subset.iloc[batch_start:batch_end]

        for idx, row in batch.iterrows():
            prompt = _build_prompt(row)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=ENRICHMENT_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            response_schema=ENRICHMENT_SCHEMA,
                        ),
                    )
                    parsed = _parse_response(response.text)
                    for col in enriched_cols:
                        if col in parsed:
                            val = parsed[col]
                            if isinstance(val, list):
                                work.at[idx, col] = json.dumps(val)
                            else:
                                work.at[idx, col] = val
                    processed += 1
                    break
                except Exception as e:
                    if attempt < max_retries - 1 and ("503" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e)):
                        time.sleep(2 ** (attempt + 1))
                        continue
                    failed += 1
                    if failed <= 10:
                        print(f"  Warning: row {idx} failed: {e}")
                    break

        done = min(batch_end, total)
        print(f"  [{done}/{total}] processed={processed} failed={failed}")

        if batch_end < total:
            if checkpoint_path:
                work.to_csv(checkpoint_path, index=False)
            time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\nEnrichment complete: {processed} ok, {failed} failed out of {total}")
    return work


def build_retrieval_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add retrieval_text column from raw + enriched fields."""
    work = df.copy()

    # Parse character_tags from JSON string back to list for building text
    def _safe_parse_tags(val: Any) -> Any:
        if isinstance(val, str) and val.startswith("["):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return val
        return val

    if "character_tags" in work.columns:
        work["_char_tags_parsed"] = work["character_tags"].apply(_safe_parse_tags)
        saved = work["character_tags"].copy()
        work["character_tags"] = work["_char_tags_parsed"]

    work["retrieval_text"] = work.apply(_build_retrieval_text, axis=1)

    if "_char_tags_parsed" in work.columns:
        work["character_tags"] = saved
        work.drop(columns=["_char_tags_parsed"], inplace=True)

    return work


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Enrich fragrance dataset with vibe attributes")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to enrich")
    parser.add_argument("--resume-from", type=int, default=0, help="Row index to resume from")
    args = parser.parse_args()

    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    # Add fragrance_id if missing
    if "fragrance_id" not in df.columns:
        df.insert(0, "fragrance_id", df.index.astype(str))
        print(f"Added fragrance_id column (0 to {len(df)-1})")

    print(f"Enriching {args.max_rows or len(df)} rows starting from {args.resume_from}...")
    enriched = enrich_dataframe(
        df,
        max_rows=args.max_rows,
        resume_from=args.resume_from,
        checkpoint_path=args.output_csv + ".ckpt",
    )

    print("Building retrieval_text...")
    enriched = build_retrieval_text(enriched)

    print(f"Saving to {args.output_csv}...")
    enriched.to_csv(args.output_csv, index=False)
    print(f"Done. Shape: {enriched.shape}")
    print(f"Sample retrieval_text:\n{enriched['retrieval_text'].iloc[args.resume_from]}")


if __name__ == "__main__":
    main()
