"""Week 2 pipeline stages -- called by notebook cells.

Every function guards torch behind runtime ``import`` statements so the module
loads on CPU-only machines for testing and linting.  numpy and pandas are
available in all environments that run this code.
"""

from __future__ import annotations

import datetime
import pathlib
import shutil
from typing import TYPE_CHECKING, Any

from vibescents.io_utils import (
    dump_json,
    ensure_dir,
    load_embeddings,
    load_json,
    save_embeddings,
)
from vibescents.similarity import normalize_rows

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
    ensure_dir(dest.parent)
    dump_json(dest, payload)


def read_manifest(path: str | pathlib.Path) -> dict[str, Any]:
    """Read and validate a ``manifest.json``.

    Raises ``ValueError`` if any of the :data:`MANIFEST_KEYS` are absent.
    """
    data: dict[str, Any] = load_json(path)
    missing = MANIFEST_KEYS - data.keys()
    if missing:
        raise ValueError(f"Manifest is missing keys: {sorted(missing)}")
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
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
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
        ``embeddings_partial_{batch_idx}.npy``.  Each checkpoint file
        contains only the *delta* rows since the previous checkpoint; use
        :func:`embed_corpus_resume` to concatenate them on recovery.
    resume_batch:
        Batch index to resume from (use :func:`embed_corpus_resume` to
        obtain the partial result and the correct index).

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), dim)``, dtype float32, L2-normalized rows.
    """
    import numpy as np  # noqa: PLC0415

    if checkpoint_dir is not None:
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        ensure_dir(checkpoint_dir)

    n_batches = (len(texts) + batch_size - 1) // batch_size
    parts: list["np.ndarray"] = []
    delta_parts: list["np.ndarray"] = []

    for batch_idx in range(resume_batch, n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]

        raw = model.embed_texts(batch)
        normalized_batch = normalize_rows(raw[:, :dim].astype(np.float32))
        parts.append(normalized_batch)
        delta_parts.append(normalized_batch)

        if (
            checkpoint_dir is not None
            and (batch_idx + 1) % _CHECKPOINT_INTERVAL == 0
        ):
            save_embeddings(
                checkpoint_dir / f"embeddings_partial_{batch_idx}.npy",
                np.vstack(delta_parts),
            )
            delta_parts.clear()
            print(
                f"  Checkpoint saved at batch {batch_idx + 1}/{n_batches} "
                f"({sum(p.shape[0] for p in parts)} rows)"
            )

    if not parts:
        return np.empty((0, dim), dtype=np.float32)

    return np.vstack(parts)


def embed_corpus_resume(
    checkpoint_dir: str | pathlib.Path,
) -> tuple["np.ndarray | None", int]:
    """Resume embedding from checkpoints.

    Globs for ``embeddings_partial_*.npy`` in *checkpoint_dir*, sorts by
    batch index, concatenates all delta files in order, and returns
    ``(partial_embeddings, next_batch_idx)``.  Each checkpoint file stores
    only the rows written since the previous checkpoint (delta format).
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

    partial = np.concatenate([load_embeddings(f) for f in files], axis=0)
    latest_batch_idx = int(files[-1].stem.split("_")[-1])
    return partial, latest_batch_idx + 1


# ---------------------------------------------------------------------------
# Embedding quality
# ---------------------------------------------------------------------------


def embedding_sanity_check(
    emb: "np.ndarray",
    n_pairs: int = 1000,
    min_variance: float = 0.001,
) -> None:
    """Spot-check embedding quality via pairwise cosine similarity variance.

    Picks *n_pairs* random row-pairs and computes their cosine similarity.
    Because *emb* is expected to be L2-normalized (as returned by
    :func:`embed_corpus`), the cosine similarity is just the dot product.
    Raises ``ValueError`` if the variance falls below *min_variance*, which
    indicates a collapsed or degenerate embedding matrix.
    """
    import numpy as np  # noqa: PLC0415

    if emb.shape[0] < 2:
        raise ValueError("Need at least 2 embeddings for a sanity check.")

    rng = np.random.default_rng(42)
    n = emb.shape[0]
    actual_pairs = min(n_pairs, n * (n - 1) // 2)

    a_idx = rng.integers(0, n, size=actual_pairs)
    b_idx = rng.integers(0, n, size=actual_pairs)
    for i in range(actual_pairs):
        max_tries = 10 * n
        tries = 0
        while a_idx[i] == b_idx[i]:
            b_idx[i] = rng.integers(0, n)
            tries += 1
            if tries >= max_tries:
                b_idx[i] = (int(a_idx[i]) + 1) % n
                break

    similarities = [float(np.dot(emb[a], emb[b])) for a, b in zip(a_idx, b_idx)]

    variance = float(np.var(similarities))
    if variance <= min_variance:
        raise ValueError(
            f"Embedding cosine-similarity variance is {variance:.6f}, "
            f"below threshold {min_variance}. "
            "The embedding matrix may be collapsed or degenerate."
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
        raise ValueError(
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
    if not result.vibe_sentence.strip():
        raise ValueError("Smoke test failed: vibe_sentence is empty.")
    if not result.likely_season:
        raise ValueError("Smoke test failed: likely_season is empty.")
    if len(result.character_tags) < 3:
        raise ValueError(
            f"Smoke test failed: character_tags has {len(result.character_tags)} items, need >= 3."
        )
    return True
