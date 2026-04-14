"""Week 2 pipeline stages -- called by notebook cells.

Every function guards heavy imports (numpy, pandas, torch) behind runtime
``import`` statements so the module loads on CPU-only machines for testing
and linting.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_KEYS = {
    "model",
    "commit_sha",
    "row_count",
    "dim",
    "created_at",
    "pipeline_version",
}


def write_manifest(
    path: str | pathlib.Path,
    *,
    model: str,
    commit_sha: str,
    row_count: int,
    dim: int,
    pipeline_version: str,
) -> None:
    """Write a ``manifest.json`` alongside an ``.npy`` artifact.

    The manifest captures provenance metadata so downstream consumers can
    verify that an artifact was produced by the expected pipeline revision.
    ``created_at`` is set to the current UTC time in ISO-8601 format.
    """
    payload = {
        "model": model,
        "commit_sha": commit_sha,
        "row_count": row_count,
        "dim": dim,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pipeline_version": pipeline_version,
    }
    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(path: str | pathlib.Path) -> dict:
    """Read and validate a ``manifest.json``.

    Raises ``AssertionError`` if any of the :data:`MANIFEST_KEYS` are absent.
    """
    text = pathlib.Path(path).read_text(encoding="utf-8")
    data: dict = json.loads(text)
    missing = MANIFEST_KEYS - data.keys()
    assert not missing, f"Manifest is missing keys: {sorted(missing)}"
    return data


# ---------------------------------------------------------------------------
# GPU + Disk
# ---------------------------------------------------------------------------


def check_disk_space(
    path: str = "/content",
    warn_pct: float = 80,
    abort_pct: float = 95,
) -> None:
    """Check disk usage at *path*.

    Prints a warning when usage exceeds *warn_pct* percent and raises
    ``RuntimeError`` when it exceeds *abort_pct* percent.  Designed for
    Colab where ``/content`` is the primary workspace.
    """
    usage = shutil.disk_usage(path)
    used_pct = (usage.used / usage.total) * 100
    if used_pct >= abort_pct:
        raise RuntimeError(
            f"Disk usage at {path} is {used_pct:.1f}% (>= {abort_pct}% abort threshold). "
            f"Free space: {usage.free / 1e9:.1f} GB."
        )
    if used_pct >= warn_pct:
        print(
            f"WARNING: Disk usage at {path} is {used_pct:.1f}% (>= {warn_pct}% warn threshold). "
            f"Free space: {usage.free / 1e9:.1f} GB."
        )


def detect_gpu_tier() -> str:
    """Return ``'A100'``, ``'L4'``, or ``'T4'`` based on VRAM.

    Requires ``torch`` with CUDA.  Raises ``RuntimeError`` if no GPU is
    available.
    """
    import torch  # noqa: PLC0415 — guarded runtime import

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")
    total_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    if total_gb >= 35:
        return "A100"
    if total_gb >= 20:
        return "L4"
    return "T4"


# ---------------------------------------------------------------------------
# Model-state tracking
# ---------------------------------------------------------------------------

_ACTIVE_MODEL: str | None = None


def get_active_model() -> str | None:
    """Return the name of the currently loaded model, or ``None``."""
    return _ACTIVE_MODEL


def ensure_model_loaded(
    target: str,
    loader_fn: Any,
    unloader_fn: Any = None,
) -> None:
    """Load *target* model if it is not already active.

    If a different model is active and *unloader_fn* is provided, the
    current model is unloaded first.  This prevents OOM on single-GPU
    Colab runtimes where only one large model fits in VRAM.
    """
    global _ACTIVE_MODEL  # noqa: PLW0603
    if _ACTIVE_MODEL == target:
        return
    if _ACTIVE_MODEL is not None and unloader_fn is not None:
        unloader_fn()
    loader_fn()
    _ACTIVE_MODEL = target


def mark_model_unloaded() -> None:
    """Clear the active-model tracker (e.g. after manual ``del model``)."""
    global _ACTIVE_MODEL  # noqa: PLW0603
    _ACTIVE_MODEL = None


# ---------------------------------------------------------------------------
# Stage gates
# ---------------------------------------------------------------------------


def stage_complete(
    stage_id: str,
    artifacts_dir: str | pathlib.Path,
    pipeline_version: str,
) -> bool:
    """Check whether *stage_id*'s manifest matches *pipeline_version*.

    Returns ``True`` only when a ``manifest.json`` exists in
    *artifacts_dir* and its ``pipeline_version`` field equals the
    requested version.  This lets the notebook skip expensive stages on
    re-run.
    """
    manifest_path = pathlib.Path(artifacts_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_manifest(manifest_path)
    return manifest.get("pipeline_version") == pipeline_version


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_CHECKPOINT_INTERVAL = 100  # batches between checkpoint writes


def embed_corpus(
    texts: list[str],
    model: Any,
    *,
    dim: int = 1024,
    batch_size: int = 32,
    checkpoint_dir: str | pathlib.Path | None = None,
    resume_batch: int = 0,
) -> "np.ndarray":
    """Embed *texts*, Matryoshka-truncate to *dim*, and L2-normalize.

    Parameters
    ----------
    texts:
        Raw retrieval-text strings to embed.
    model:
        An embedder instance exposing ``embed_texts(batch) -> np.ndarray``
        (e.g. :class:`~vibescents.embeddings.VoyageEmbedder`).
    dim:
        Target dimensionality after Matryoshka truncation.
    batch_size:
        Number of texts per embedding call.
    checkpoint_dir:
        If set, partial embeddings are saved every
        :data:`_CHECKPOINT_INTERVAL` batches as
        ``embeddings_partial_{batch_idx}.npy``.
    resume_batch:
        Batch index to resume from (use :func:`embed_corpus_resume` to
        obtain partial results and the correct index).

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), dim)``, dtype float32, L2-normalized rows.
    """
    import numpy as np  # noqa: PLC0415

    if checkpoint_dir is not None:
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_batches = (len(texts) + batch_size - 1) // batch_size
    parts: list["np.ndarray"] = []

    for batch_idx in range(resume_batch, n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]

        raw = model.embed_texts(batch)
        truncated = raw[:, :dim]
        parts.append(truncated)

        # Checkpoint every _CHECKPOINT_INTERVAL batches
        if (
            checkpoint_dir is not None
            and (batch_idx + 1) % _CHECKPOINT_INTERVAL == 0
        ):
            partial = np.vstack(parts)
            np.save(
                checkpoint_dir / f"embeddings_partial_{batch_idx}.npy",
                partial.astype(np.float32),
            )
            print(
                f"  Checkpoint saved at batch {batch_idx + 1}/{n_batches} "
                f"({partial.shape[0]} rows)"
            )

    if not parts:
        return np.empty((0, dim), dtype=np.float32)

    matrix = np.vstack(parts).astype(np.float32)

    # L2-normalize each row
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    matrix = matrix / norms

    return matrix


def embed_corpus_resume(
    checkpoint_dir: str | pathlib.Path,
) -> tuple["np.ndarray | None", int]:
    """Resume embedding from the latest checkpoint.

    Globs for ``embeddings_partial_*.npy`` in *checkpoint_dir*, sorts by
    batch index, and returns ``(partial_embeddings, next_batch_idx)``.
    If no checkpoints exist, returns ``(None, 0)``.
    """
    import numpy as np  # noqa: PLC0415

    ckpt_path = pathlib.Path(checkpoint_dir)
    files = sorted(
        ckpt_path.glob("embeddings_partial_*.npy"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not files:
        return None, 0

    # The latest checkpoint file contains all rows up to that point
    latest = files[-1]
    batch_idx = int(latest.stem.split("_")[-1])
    partial = np.load(latest)
    return partial, batch_idx + 1


# ---------------------------------------------------------------------------
# Embedding quality
# ---------------------------------------------------------------------------


def embedding_sanity_check(
    emb: "np.ndarray",
    n_pairs: int = 10,
    min_variance: float = 0.01,
) -> None:
    """Spot-check embedding quality via pairwise cosine similarity variance.

    Picks *n_pairs* random row-pairs, computes their cosine similarity,
    and asserts that the variance exceeds *min_variance*.  A collapsed or
    degenerate embedding matrix (all rows identical) will have near-zero
    variance and fail this check.
    """
    import numpy as np  # noqa: PLC0415

    if emb.shape[0] < 2:
        raise ValueError("Need at least 2 embeddings for a sanity check.")

    rng = np.random.default_rng(42)
    n = emb.shape[0]
    actual_pairs = min(n_pairs, n * (n - 1) // 2)

    indices = rng.choice(n, size=(actual_pairs, 2), replace=True)
    # Re-draw any self-pairs
    for i in range(actual_pairs):
        while indices[i, 0] == indices[i, 1]:
            indices[i, 1] = rng.integers(0, n)

    similarities: list[float] = []
    for a, b in indices:
        dot = float(np.dot(emb[a], emb[b]))
        norm_a = float(np.linalg.norm(emb[a]))
        norm_b = float(np.linalg.norm(emb[b]))
        denom = norm_a * norm_b
        cos = dot / denom if denom > 0 else 0.0
        similarities.append(cos)

    variance = float(np.var(similarities))
    assert variance > min_variance, (
        f"Embedding cosine-similarity variance is {variance:.6f}, "
        f"below threshold {min_variance}. "
        f"The embedding matrix may be collapsed or degenerate."
    )


# ---------------------------------------------------------------------------
# TIER B selection
# ---------------------------------------------------------------------------


def select_tier_b(
    df: "pd.DataFrame",
    target_size: int = 2000,
    min_size: int = 500,
) -> "pd.DataFrame":
    """Select top rows by ``rating_count`` with complete metadata.

    **Strict filter** (preferred): requires ``top_notes``, ``middle_notes``,
    ``base_notes``, and ``main_accords`` all non-null.  Rows passing the
    filter are sorted by ``rating_count`` descending and the top
    *target_size* are returned.

    **Fallback filter**: if the strict filter yields fewer than
    *target_size* rows, relax to require only ``top_notes`` and
    ``main_accords``.

    **Hard minimum**: raises ``ValueError`` if even the relaxed filter
    yields fewer than *min_size* rows.
    """
    import pandas as pd  # noqa: PLC0415

    strict_cols = ["top_notes", "middle_notes", "base_notes", "main_accords"]
    relaxed_cols = ["top_notes", "main_accords"]

    mask_strict = df[strict_cols].notna().all(axis=1)
    strict_df = df.loc[mask_strict]

    if len(strict_df) >= target_size:
        selected = strict_df.sort_values("rating_count", ascending=False).head(
            target_size
        )
        print(
            f"Tier B: {len(selected)} rows selected (strict filter, "
            f"{len(strict_df)} candidates)."
        )
        return selected.reset_index(drop=True)

    # Fallback: relax filter
    mask_relaxed = df[relaxed_cols].notna().all(axis=1)
    relaxed_df = df.loc[mask_relaxed]

    if len(relaxed_df) < min_size:
        raise ValueError(
            f"Even the relaxed filter yields only {len(relaxed_df)} rows, "
            f"below the hard minimum of {min_size}. "
            "The source dataset may be too sparse for Tier B."
        )

    selected = relaxed_df.sort_values("rating_count", ascending=False).head(
        target_size
    )
    print(
        f"Tier B: {len(selected)} rows selected (relaxed filter, "
        f"{len(relaxed_df)} candidates)."
    )
    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Enrichment validation
# ---------------------------------------------------------------------------


def validate_enrichment(
    df: "pd.DataFrame",
    min_success_rate: float = 0.98,
) -> None:
    """Assert that at least *min_success_rate* of rows have a non-null ``vibe_sentence``.

    This is a post-enrichment gate: if too many rows failed enrichment,
    the embedding stage would produce a low-quality corpus.
    """
    total = len(df)
    if total == 0:
        raise ValueError("Enrichment dataframe is empty.")
    success = int(df["vibe_sentence"].notna().sum())
    rate = success / total
    if rate < min_success_rate:
        raise AssertionError(
            f"Enrichment success rate {rate:.4f} ({success}/{total}) "
            f"is below the required {min_success_rate:.2%} threshold."
        )
    print(
        f"Enrichment validation passed: {rate:.2%} success "
        f"({success}/{total} rows)."
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test_enrichment(
    client: Any,
    sample_row: "pd.Series",
) -> bool:
    """Run a single-row enrichment and validate against :class:`EnrichmentSchemaV2`.

    Returns ``True`` if the enrichment succeeds and the result validates.
    Raises on any failure so the notebook can surface the error early
    before committing to a full enrichment run.
    """
    from vibescents.enrich import _build_prompt  # noqa: PLC0415
    from vibescents.schemas import EnrichmentSchemaV2  # noqa: PLC0415

    prompt = _build_prompt(sample_row)
    result = client.generate(prompt)
    if not isinstance(result, EnrichmentSchemaV2):
        raise TypeError(
            f"Expected EnrichmentSchemaV2, got {type(result).__name__}."
        )
    # Validate required fields are populated
    assert result.vibe_sentence.strip(), "vibe_sentence is empty"
    assert result.likely_season, "likely_season is empty"
    assert len(result.character_tags) >= 3, (
        f"character_tags has {len(result.character_tags)} items, need >= 3"
    )
    return True
