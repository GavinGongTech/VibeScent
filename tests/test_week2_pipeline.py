from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import vibescents.week2_pipeline as w2


# ---------------------------------------------------------------------------
# Torch stub — torch is an optional GPU dependency not installed in CI.
# We inject a minimal fake module into sys.modules so unittest.mock.patch
# can resolve "torch.cuda.*" targets without a real torch installation.
# ---------------------------------------------------------------------------


def _make_torch_stub() -> types.ModuleType:
    torch_mod = types.ModuleType("torch")
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = MagicMock(return_value=True)
    cuda_mod.get_device_properties = MagicMock()
    torch_mod.cuda = cuda_mod
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.cuda", cuda_mod)
    return torch_mod


_TORCH_STUB = _make_torch_stub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest_to_dir(
    artifacts_dir: Path,
    *,
    pipeline_version: str = "v2",
) -> None:
    """Write a valid manifest.json into *artifacts_dir*, matching stage_complete's expected path."""
    w2.write_manifest(
        artifacts_dir / "manifest.json",
        model="voyage-3-large",
        commit_sha="abc1234",
        row_count=5000,
        dim=1024,
        pipeline_version=pipeline_version,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_write_read_manifest_roundtrips_all_keys(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    w2.write_manifest(
        manifest_path,
        model="voyage-3-large",
        commit_sha="abc1234",
        row_count=5000,
        dim=1024,
        pipeline_version="v2",
    )
    result = w2.read_manifest(manifest_path)
    assert set(result.keys()) == w2.MANIFEST_KEYS
    assert result["model"] == "voyage-3-large"
    assert result["commit_sha"] == "abc1234"
    assert result["row_count"] == 5000
    assert result["dim"] == 1024
    assert result["pipeline_version"] == "v2"
    assert "created_at" in result


def test_read_manifest_missing_keys_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    # Write a file that lacks most required keys.
    manifest_path.write_text(json.dumps({"model": "voyage-3-large", "dim": 1024}))
    with pytest.raises(ValueError):
        w2.read_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Stage gates
# ---------------------------------------------------------------------------


def test_stage_complete_true_when_manifest_version_matches(tmp_path: Path) -> None:
    _write_manifest_to_dir(tmp_path, pipeline_version="v2")
    assert w2.stage_complete("embed", tmp_path, "v2") is True


def test_stage_complete_false_on_pipeline_version_mismatch(tmp_path: Path) -> None:
    _write_manifest_to_dir(tmp_path, pipeline_version="v1")
    assert w2.stage_complete("embed", tmp_path, "v2") is False


def test_stage_complete_false_when_no_manifest_exists(tmp_path: Path) -> None:
    assert w2.stage_complete("embed", tmp_path, "v2") is False


# ---------------------------------------------------------------------------
# GPU tier detection
# ---------------------------------------------------------------------------


def test_detect_gpu_tier_returns_a100_for_40gb_plus() -> None:
    props = MagicMock()
    props.total_memory = 45 * 1_000_000_000  # 45 GB (uses 1e9 divisor)
    # Patch via the stub already in sys.modules so the runtime import in the
    # function under test resolves to the same object.
    _TORCH_STUB.cuda.is_available = MagicMock(return_value=True)
    _TORCH_STUB.cuda.get_device_properties = MagicMock(return_value=props)
    assert w2.detect_gpu_tier() == "A100"


def test_detect_gpu_tier_returns_l4_for_mid_range() -> None:
    props = MagicMock()
    props.total_memory = 22 * 1_000_000_000  # 22 GB
    _TORCH_STUB.cuda.is_available = MagicMock(return_value=True)
    _TORCH_STUB.cuda.get_device_properties = MagicMock(return_value=props)
    assert w2.detect_gpu_tier() == "L4"


def test_detect_gpu_tier_returns_t4_for_small() -> None:
    props = MagicMock()
    props.total_memory = 14 * 1_000_000_000  # 14 GB
    _TORCH_STUB.cuda.is_available = MagicMock(return_value=True)
    _TORCH_STUB.cuda.get_device_properties = MagicMock(return_value=props)
    assert w2.detect_gpu_tier() == "T4"


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------


def _fake_disk_usage(total_bytes: int, used_pct: float):
    import collections

    DiskUsage = collections.namedtuple("DiskUsage", ["total", "used", "free"])
    used = int(total_bytes * used_pct / 100)
    return DiskUsage(total=total_bytes, used=used, free=total_bytes - used)


def test_check_disk_space_abort_raises_when_above_abort_pct() -> None:
    usage = _fake_disk_usage(100 * 1024**3, used_pct=96)
    with patch("shutil.disk_usage", return_value=usage):
        with pytest.raises(RuntimeError):
            w2.check_disk_space(path="/content", warn_pct=80, abort_pct=95)


def test_check_disk_space_ok_does_not_raise_when_below_warn_pct() -> None:
    usage = _fake_disk_usage(100 * 1024**3, used_pct=50)
    with patch("shutil.disk_usage", return_value=usage):
        w2.check_disk_space(path="/content", warn_pct=80, abort_pct=95)  # must not raise


# ---------------------------------------------------------------------------
# Model state tracking
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_model_state() -> None:
    """Ensure global model state is clean before and after every test."""
    w2.mark_model_unloaded()
    yield
    w2.mark_model_unloaded()


def test_model_state_tracking_full_lifecycle() -> None:
    loader_a = MagicMock()
    loader_b = MagicMock()
    unloader_a = MagicMock()

    # Initially no model is loaded.
    assert w2.get_active_model() is None

    # Load model A — loader_a must be called exactly once.
    w2.ensure_model_loaded("model-a", loader_fn=loader_a, unloader_fn=unloader_a)
    loader_a.assert_called_once()
    assert w2.get_active_model() == "model-a"

    # Request model A again — loader must NOT be called a second time.
    w2.ensure_model_loaded("model-a", loader_fn=loader_a, unloader_fn=unloader_a)
    loader_a.assert_called_once()  # count is still 1

    # Switch to model B — pass unloader_a so it gets called, then loader_b fires.
    w2.ensure_model_loaded("model-b", loader_fn=loader_b, unloader_fn=unloader_a)
    unloader_a.assert_called_once()
    loader_b.assert_called_once()
    assert w2.get_active_model() == "model-b"


# ---------------------------------------------------------------------------
# Embedding sanity check
# ---------------------------------------------------------------------------


def test_embedding_sanity_check_pass_with_random_normalized_array() -> None:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((100, 1024)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms  # unit-normalize
    # Random unit vectors in 1024-d produce cosine-similarity variance ~0.001
    # with only 10 pairs, which is far below the default 0.01 threshold.
    # Use a threshold that a legitimate diverse embedding surpasses but an
    # all-identical embedding (variance=0) cannot.
    w2.embedding_sanity_check(emb, n_pairs=10, min_variance=0.0001)  # must not raise


def test_embedding_sanity_check_fail_when_all_rows_identical() -> None:
    row = np.ones((1, 1024), dtype=np.float32)
    row /= np.linalg.norm(row)
    emb = np.tile(row, (100, 1))  # all rows identical → variance exactly 0.0
    with pytest.raises((AssertionError, ValueError)):
        # Even a very lenient threshold rejects zero-variance embeddings.
        w2.embedding_sanity_check(emb, n_pairs=10, min_variance=0.0001)


# ---------------------------------------------------------------------------
# Tier B selection
# ---------------------------------------------------------------------------


def _make_df(
    n: int,
    *,
    fill_notes: bool = True,
    fill_relaxed_only: bool = False,
    extra_empty: int = 0,
) -> pd.DataFrame:
    """
    Build a synthetic fragrance DataFrame.

    Parameters
    ----------
    n:
        Total number of rows.
    fill_notes:
        If True, all four note columns are non-null (passes strict filter).
    fill_relaxed_only:
        If True, only top_notes and main_accords are non-null (passes relaxed
        filter but not strict).
    extra_empty:
        Number of rows from the end where all note columns are null.
    """
    base = {
        "fragrance_id": [str(i) for i in range(n)],
        "rating_count": list(range(n, 0, -1)),
        "top_notes": [f"top_{i}" if (fill_notes or fill_relaxed_only) else None for i in range(n)],
        "middle_notes": [f"mid_{i}" if fill_notes else None for i in range(n)],
        "base_notes": [f"base_{i}" if fill_notes else None for i in range(n)],
        "main_accords": [f"accord_{i}" if (fill_notes or fill_relaxed_only) else None for i in range(n)],
    }
    df = pd.DataFrame(base)
    if extra_empty > 0:
        for col in ("top_notes", "middle_notes", "base_notes", "main_accords"):
            df.loc[n - extra_empty :, col] = None
    return df


def test_select_tier_b_strict_returns_exactly_target_size() -> None:
    # 3000 rows all with full notes — strict path should return exactly 2000.
    df = _make_df(3000, fill_notes=True)
    result = w2.select_tier_b(df, target_size=2000, min_size=500)
    assert len(result) == 2000


def test_select_tier_b_fallback_returns_rows_from_relaxed_filter() -> None:
    # 100 fully-filled rows + 500 relaxed-only rows + 1400 empty rows.
    # Strict filter yields 100 (< 2000), so code falls to relaxed filter.
    # Relaxed filter (top_notes + main_accords) matches 600 rows (>= min_size=500).
    full = _make_df(100, fill_notes=True)
    relaxed = _make_df(500, fill_relaxed_only=True)
    relaxed["fragrance_id"] = [str(i + 100) for i in range(500)]
    relaxed["rating_count"] = list(range(500, 0, -1))
    empty = _make_df(1400, fill_notes=False)
    empty["fragrance_id"] = [str(i + 600) for i in range(1400)]
    empty["rating_count"] = list(range(1400, 0, -1))
    df = pd.concat([full, relaxed, empty], ignore_index=True)

    result = w2.select_tier_b(df, target_size=2000, min_size=500)
    assert len(result) >= 500


def test_select_tier_b_too_few_raises() -> None:
    # Only 10 rows with any data — well below min_size=500.
    df = _make_df(10, fill_notes=True)
    with pytest.raises(ValueError):
        w2.select_tier_b(df, target_size=2000, min_size=500)


# ---------------------------------------------------------------------------
# Enrichment validation
# ---------------------------------------------------------------------------


def _enrichment_df(n: int, n_valid_vibe: int) -> pd.DataFrame:
    vibe = ["A vibe sentence."] * n_valid_vibe + [None] * (n - n_valid_vibe)
    return pd.DataFrame({"vibe_sentence": vibe})


def test_validate_enrichment_pass_at_high_success_rate() -> None:
    df = _enrichment_df(500, 495)  # 495/500 = 99% — above 98% threshold
    w2.validate_enrichment(df, min_success_rate=0.98)  # must not raise


def test_validate_enrichment_fail_below_min_success_rate() -> None:
    df = _enrichment_df(500, 480)  # 480/500 = 96% — below 98% threshold
    with pytest.raises((AssertionError, ValueError)):
        w2.validate_enrichment(df, min_success_rate=0.98)


# ---------------------------------------------------------------------------
# embed_corpus_resume
# ---------------------------------------------------------------------------


def test_embed_corpus_resume_returns_none_and_zero_for_empty_dir(tmp_path: Path) -> None:
    result, next_batch = w2.embed_corpus_resume(tmp_path)
    assert result is None
    assert next_batch == 0


def test_embed_corpus_resume_with_checkpoints_returns_concatenated_deltas(
    tmp_path: Path,
) -> None:
    # Each checkpoint file contains only the delta rows since the previous
    # checkpoint (not a cumulative snapshot).  embed_corpus_resume concatenates
    # all files in batch-index order to reconstruct the full partial result.
    batch0 = np.random.default_rng(0).standard_normal((50, 1024)).astype(np.float32)
    batch1 = np.random.default_rng(1).standard_normal((50, 1024)).astype(np.float32)

    np.save(tmp_path / "embeddings_partial_0.npy", batch0)
    np.save(tmp_path / "embeddings_partial_1.npy", batch1)

    result, next_batch = w2.embed_corpus_resume(tmp_path)

    assert result is not None
    assert next_batch == 2
    expected = np.concatenate([batch0, batch1], axis=0)
    assert result.shape == expected.shape
    assert np.allclose(result, expected)
