from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vibescents.embeddings import Qwen3VLMultimodalEmbedder
from vibescents.io_utils import dump_json, ensure_dir, save_dataframe, save_embeddings
from vibescents.schemas import RetrievalCandidate
from vibescents.similarity import (
    cosine_similarity_matrix,
    normalize_rows,
    top_k_indices,
)


def embed_text_frame(
    frame: pd.DataFrame,
    *,
    id_column: str,
    text_column: str,
    output_dir: str | Path,
    model: str | None = None,
    input_type: str = "document",
) -> np.ndarray:
    embedder = Qwen3VLMultimodalEmbedder()
    output_path = ensure_dir(Path(output_dir))
    texts = frame[text_column].fillna("").astype(str).tolist()
    embeddings = embedder.embed_multimodal_documents(texts)
    metadata = frame[[id_column, text_column]].copy()
    save_embeddings(output_path / "embeddings.npy", embeddings)
    save_dataframe(output_path / "metadata.csv", metadata)
    return embeddings


def embed_occasions(
    occasions: list[dict[str, str]],
    *,
    output_dir: str | Path,
) -> np.ndarray:
    output_path = ensure_dir(Path(output_dir))
    embedder = Qwen3VLMultimodalEmbedder()
    texts = [item["text"] for item in occasions]
    ids = [item["occasion_id"] for item in occasions]
    embeddings = embedder.embed_multimodal_documents(texts)
    similarity = cosine_similarity_matrix(embeddings)

    metadata = pd.DataFrame({"occasion_id": ids, "text": texts})
    matrix = pd.DataFrame(similarity, index=ids, columns=ids)

    save_embeddings(output_path / "embeddings.npy", embeddings)
    save_dataframe(output_path / "metadata.csv", metadata)
    save_dataframe(
        output_path / "similarity.csv", matrix.reset_index(names="occasion_id")
    )
    _save_heatmap(matrix.to_numpy(), ids, output_path / "similarity_heatmap.png")
    return embeddings


def retrieve_with_multimodal_query(
    frame: pd.DataFrame,
    *,
    id_column: str,
    text_column: str,
    occasion_text: str,
    image_path: str | Path | None,
    output_dir: str | Path,
    top_k: int = 10,
) -> list[RetrievalCandidate]:
    output_path = ensure_dir(Path(output_dir))
    embedder = Qwen3VLMultimodalEmbedder()
    doc_embeddings = embedder.embed_multimodal_documents(
        frame[text_column].fillna("").astype(str).tolist()
    )
    query_embedding = embedder.embed_multimodal_query(
        text=occasion_text, image_path=image_path
    )
    scores = cosine_similarity_matrix(query_embedding, doc_embeddings)[0]
    selected = top_k_indices(scores, top_k)

    scored = frame.copy()
    scored["multimodal_score"] = scores
    top_rows = scored.iloc[selected].copy()
    save_embeddings(output_path / "document_embeddings.npy", doc_embeddings)
    save_embeddings(output_path / "query_embedding.npy", query_embedding)
    save_dataframe(output_path / "all_scores.csv", scored)

    candidates = [
        RetrievalCandidate(
            fragrance_id=str(row[id_column]),
            name=row["name"] if "name" in top_rows.columns else None,
            brand=row["brand"] if "brand" in top_rows.columns else None,
            retrieval_text=str(row[text_column]),
            display_text=row["display_text"]
            if "display_text" in top_rows.columns
            else None,
            baseline_score=float(row["multimodal_score"]),
            metadata={
                key: _python_scalar(row[key])
                for key in top_rows.columns
                if key not in {text_column, "display_text"}
            },
        )
        for _, row in top_rows.iterrows()
    ]
    dump_json(
        output_path / "top_candidates.json",
        [candidate.model_dump() for candidate in candidates],
    )
    return candidates


def _save_heatmap(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".mplconfig"))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _python_scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def load_karans_embeddings(
    source_path: str | Path,
    *,
    output_dim: int = 1024,
    expected_rows: int | None = None,
    l2_normalize: bool = True,
) -> np.ndarray:
    """Load Karan's embedding matrix and optionally Matryoshka-truncate it."""
    matrix = np.load(Path(source_path))
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D embedding matrix, got shape {matrix.shape}.")
    rows, dims = matrix.shape
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {rows}.")
    if dims < output_dim:
        raise ValueError(
            f"Cannot truncate to {output_dim} dimensions from {dims}. "
            "Re-embed with a higher-dimensional model."
        )

    projected = matrix[:, :output_dim].astype(np.float32, copy=False)
    if l2_normalize:
        projected = normalize_rows(projected)
    return projected
