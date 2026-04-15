"""Step 4 — Text-to-fragrance retrieval comparison: RAW vs ENRICHED.

Loads pre-computed occasion embeddings and both fragrance embedding matrices,
computes cosine similarity, and prints side-by-side top-5 results per occasion.

Usage (from project root):
    uv run python scripts/compare_retrieval.py

Outputs:
    artifacts/retrieval_comparison.txt  — full comparison report
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from vibescents.similarity import cosine_similarity_matrix, top_k_indices

ARTIFACTS = Path("artifacts")
TOP_K = 5

CORPORA = [
    (
        "RAW (embedding_text — note concatenation only)",
        ARTIFACTS / "fragrance_raw" / "embeddings.npy",
        ARTIFACTS / "fragrance_raw" / "metadata.csv",
    ),
    (
        "ENRICHED (retrieval_text — with Season/Occasion/Formality/Vibe)",
        ARTIFACTS / "fragrance_enriched" / "embeddings.npy",
        ARTIFACTS / "fragrance_enriched" / "metadata.csv",
    ),
]


def _check_artifacts() -> bool:
    missing = []
    required = [ARTIFACTS / "occasions" / "embeddings.npy", ARTIFACTS / "occasions" / "metadata.csv"]
    for _, emb, meta in CORPORA:
        required += [emb, meta]
    for path in required:
        if not path.exists():
            missing.append(str(path))
    if missing:
        print("Missing artifacts — run the embedding pipeline first:")
        for m in missing:
            print(f"  {m}")
        return False
    return True


def run_comparison(output_file: Path | None = None) -> None:
    if not _check_artifacts():
        sys.exit(1)

    occasion_emb = np.load(ARTIFACTS / "occasions" / "embeddings.npy")
    occasions_meta = pd.read_csv(ARTIFACTS / "occasions" / "metadata.csv")

    lines: list[str] = []

    for label, emb_path, meta_path in CORPORA:
        frag_emb = np.load(emb_path)
        frag_meta = pd.read_csv(meta_path)
        scores = cosine_similarity_matrix(occasion_emb, frag_emb)

        lines.append(f"\n{'=' * 70}")
        lines.append(label)
        lines.append(f"Fragrance embedding shape: {frag_emb.shape}")
        lines.append("=" * 70)

        for i, occ_row in occasions_meta.iterrows():
            occ_id = occ_row["occasion_id"]
            top_idx = top_k_indices(scores[i], TOP_K)
            lines.append(f"\n--- {occ_id} ---")
            for rank, j in enumerate(top_idx, 1):
                name = frag_meta.iloc[j].get("name", frag_meta.iloc[j].iloc[1])
                brand = frag_meta.iloc[j].get("brand", "")
                score = scores[i, j]
                lines.append(f"  {rank}. {score:.4f}  {brand} — {name}")

    # Score-change summary: how different are the top-5 sets between RAW and ENRICHED?
    lines.append(f"\n\n{'=' * 70}")
    lines.append("OVERLAP ANALYSIS: RAW vs ENRICHED top-5 sets per occasion")
    lines.append("=" * 70)

    raw_emb = np.load(CORPORA[0][1])
    raw_meta = pd.read_csv(CORPORA[0][2])
    enr_emb = np.load(CORPORA[1][1])
    enr_meta = pd.read_csv(CORPORA[1][2])

    raw_scores = cosine_similarity_matrix(occasion_emb, raw_emb)
    enr_scores = cosine_similarity_matrix(occasion_emb, enr_emb)

    total_overlap = 0
    for i, occ_row in occasions_meta.iterrows():
        occ_id = occ_row["occasion_id"]
        raw_names = set(str(raw_meta.iloc[j].get("name", "")) for j in top_k_indices(raw_scores[i], TOP_K))
        enr_names = set(str(enr_meta.iloc[j].get("name", "")) for j in top_k_indices(enr_scores[i], TOP_K))
        shared = len(raw_names & enr_names)
        total_overlap += shared
        lines.append(f"  {occ_id}: {shared}/{TOP_K} shared  (changed: {TOP_K - shared})")
    avg_overlap = total_overlap / len(occasions_meta)
    lines.append(f"\n  Average overlap: {avg_overlap:.1f}/{TOP_K} — {'SIMILAR' if avg_overlap > 3 else 'DIVERGENT'} retrieval sets")

    report = "\n".join(lines)
    print(report)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report, encoding="utf-8")
        print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    run_comparison(output_file=ARTIFACTS / "retrieval_comparison.txt")
