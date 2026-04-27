#!/usr/bin/env python3
"""Enrich vibescent_enriched.csv using Groq free API.

Runs enrichment on the most popular fragrances first (sorted by rating_count),
so the highest-value rows are done even if the run is interrupted.

Usage:
    export GROQ_API_KEY=gsk_...
    uv run python -m scripts.enrich_groq

Resumes automatically from checkpoint on restart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from vibescents.enrich import (
    GroqEnrichmentClient,
    enrich_dataframe,
    build_retrieval_text,
)

CSV_IN = ROOT / "data" / "vibescent_enriched.csv"
CSV_OUT = ROOT / "data" / "vibescent_enriched.csv"  # overwrite in-place
CKPT = ROOT / "data" / "vibescent_enriched.csv.ckpt"
FAILS = ROOT / "data" / "enrichment_failures.jsonl"

# Batch size 1 — Groq is sequential, rate limiter handles pacing
BATCH_SIZE = 1


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set.")
        print("  Sign up free at https://console.groq.com")
        print("  Then: export GROQ_API_KEY=gsk_...")
        sys.exit(1)

    print(f"Loading {CSV_IN} ...")
    df = pd.read_csv(CSV_IN, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    # Resume from checkpoint if available
    if CKPT.exists():
        print(f"  Checkpoint found: {CKPT} — merging ...")
        ckpt = pd.read_csv(CKPT, low_memory=False)
        for col in ckpt.columns:
            if col not in df.columns:
                df[col] = ckpt[col]
        df.update(ckpt)

    # Sort so most popular fragrances are enriched first
    if "rating_count" in df.columns:
        df["_sort_key"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)
        df = df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])
        df = df.reset_index(drop=True)

    done = df["vibe_sentence"].notna().sum() if "vibe_sentence" in df.columns else 0
    todo = len(df) - done
    print(f"  Already enriched: {done:,} / {len(df):,}")
    print(f"  Remaining       : {todo:,}")
    eta_hours = todo / 14 / 60
    print(f"  ETA             : {eta_hours:.1f} hours at ~14 rows/min (Groq free tier)")

    if todo == 0:
        print("All rows already enriched. Building retrieval_text and saving ...")
        df = build_retrieval_text(df)
        df.to_csv(CSV_OUT, index=False)
        print(f"Saved: {CSV_OUT}")
        return

    client = GroqEnrichmentClient()
    print(f"\nStarting enrichment with {client.model_name} ...")
    print("  Press Ctrl+C to stop — progress is saved every row via checkpoint.\n")

    enriched = enrich_dataframe(
        df,
        client=client,
        batch_size=BATCH_SIZE,
        checkpoint_path=str(CKPT),
        failures_path=str(FAILS),
    )

    enriched = build_retrieval_text(enriched)
    enriched.to_csv(CSV_OUT, index=False)
    print(f"\nSaved enriched CSV: {CSV_OUT}")
    print(f"Checkpoint kept at: {CKPT}")


if __name__ == "__main__":
    main()
