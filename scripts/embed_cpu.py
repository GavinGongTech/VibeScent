#!/usr/bin/env python3
"""Embed vibescent_enriched.csv using nomic-embed-text-v1.5 (CPU, ~30s for 35k rows).

Run this to get a working corpus immediately. Re-run after enrichment completes
to pick up the richer retrieval_text.

Usage:
    uv run python -m scripts.embed_cpu
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from vibescents.embeddings import SentenceTransformerEmbedder
from vibescents.enrich import build_retrieval_text

CSV_IN  = ROOT / "data" / "vibescent_enriched.csv"
OUT_DIR = ROOT / "artifacts" / "qwen3vl_corpus"
EMB_OUT = OUT_DIR / "embeddings.npy"
MAN_OUT = OUT_DIR / "manifest.json"


def main() -> None:
    print(f"Loading {CSV_IN} ...")
    df = pd.read_csv(CSV_IN, low_memory=False)
    print(f"  {len(df):,} rows")

    if "retrieval_text" not in df.columns or df["retrieval_text"].isna().all():
        print("  Building retrieval_text from raw fields ...")
        df = build_retrieval_text(df)

    texts = df["retrieval_text"].fillna(df.get("name", "")).tolist()
    print(f"  Sample: {texts[0][:100]}")

    print("\nLoading nomic-embed-text-v1.5 ...")
    embedder = SentenceTransformerEmbedder()

    print(f"Embedding {len(texts):,} texts ...")
    embeddings = embedder.embed_multimodal_documents(texts, batch_size=256)
    print(f"  Shape: {embeddings.shape}")

    assert embeddings.shape[0] == len(texts)
    assert np.isnan(embeddings).sum() == 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_OUT, embeddings)

    manifest = {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "count": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "l2_normalized": True,
        "note": "CPU embedding — multimodal channel disabled at query time",
    }
    MAN_OUT.write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved: {EMB_OUT}")
    print(f"Manifest: {MAN_OUT}")
    print("\nNext: run ./start.sh to verify the backend loads correctly.")


if __name__ == "__main__":
    main()
