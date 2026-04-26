from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from vibescents.image_scoring import (
    ImageHeadProbabilities,
    NeilCNNWrapper,
    discretize_day_night,
    discretize_formality,
    discretize_gender,
    discretize_frequency,
    image_negative_log_likelihood,
    score_candidate_pool,
    season_target_index,
)


def test_discretize_formality_and_time_buckets() -> None:
    assert discretize_formality(0.0) == 0
    assert discretize_formality(0.5) == 1
    assert discretize_formality(0.9) == 2
    assert discretize_day_night(0.49) == 0
    assert discretize_day_night(0.50) == 1


def test_discretize_gender_and_frequency() -> None:
    assert discretize_gender("male") == 0
    assert discretize_gender("female") == 1
    assert discretize_gender("neutral") == 2
    assert discretize_gender(None) == 2       # defaults to neutral
    assert discretize_frequency("occasional") == 0
    assert discretize_frequency("everyday") == 1
    assert discretize_frequency(None) == 1    # defaults to everyday


def test_season_target_index_uses_argmax_for_all_season() -> None:
    probs = np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32)
    assert season_target_index("all-season", probs) == 2
    assert season_target_index("winter", probs) == 3


def test_image_negative_log_likelihood_matches_expected_value() -> None:
    head_probabilities = {
        "formal": np.array([0.8, 0.1, 0.1], dtype=np.float32),
        "season": np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
        "time": np.array([0.9, 0.1], dtype=np.float32),
        "gender": np.array([0.6, 0.3, 0.1], dtype=np.float32),
        "frequency": np.array([0.3, 0.7], dtype=np.float32),
    }
    fragrance = {
        "formality": 0.2,
        "day_night": 0.2,
        "likely_season": "spring",
        "gender": "male",
        "frequency": "everyday",
    }
    nll = image_negative_log_likelihood(head_probabilities, fragrance)
    # formal→idx0=0.8, season→spring=0.7, time→day=0.9, gender→male=0.6, frequency→everyday=0.7
    expected = -(math.log(0.8) + math.log(0.7) + math.log(0.9) + math.log(0.6) + math.log(0.7))
    assert math.isclose(nll, expected, rel_tol=1e-6)


def test_score_candidate_pool_returns_unit_range_values() -> None:
    head_probabilities = {
        "formal": np.array([0.8, 0.1, 0.1], dtype=np.float32),
        "season": np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
        "time": np.array([0.9, 0.1], dtype=np.float32),
        "gender": np.array([0.6, 0.3, 0.1], dtype=np.float32),
        "frequency": np.array([0.3, 0.7], dtype=np.float32),
    }
    candidates = [
        {"formality": 0.2, "day_night": 0.2, "likely_season": "spring", "gender": "male", "frequency": "everyday"},
        {"formality": 0.8, "day_night": 0.8, "likely_season": "winter", "gender": "female", "frequency": "occasional"},
    ]
    scores = score_candidate_pool(head_probabilities, candidates)
    assert scores.shape == (2,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def _build_torch_stub(state_dict: dict) -> types.ModuleType:
    torch_mod = types.ModuleType("torch")
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = MagicMock(return_value=False)
    torch_mod.cuda = cuda_mod
    torch_mod.no_grad = MagicMock(
        return_value=MagicMock(
            __enter__=lambda s, *a: None, __exit__=lambda s, *a: None
        )
    )
    torch_mod.load = MagicMock(return_value=state_dict)
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.cuda", cuda_mod)
    return torch_mod


def test_from_checkpoint_loads_neil_model_state_dict_key(tmp_path: Path) -> None:
    """NeilCNNWrapper.from_checkpoint must unwrap Neil's 'model_state_dict' key."""
    fake_weights = {"weight": MagicMock()}
    torch_stub = _build_torch_stub({"model_state_dict": fake_weights})

    fake_model = MagicMock()

    # Patch torch.load to return Neil's format
    torch_stub.load = MagicMock(return_value={"model_state_dict": fake_weights})
    torch_stub.cuda.is_available = MagicMock(return_value=False)

    import unittest.mock as mock

    with mock.patch.dict(
        sys.modules, {"torch": torch_stub, "torch.cuda": torch_stub.cuda}
    ):
        wrapper = NeilCNNWrapper.from_checkpoint(
            tmp_path / "best.pt",
            model_builder=lambda: fake_model,
        )
    fake_model.load_state_dict.assert_called_once_with(fake_weights)
    assert isinstance(wrapper, NeilCNNWrapper)


def test_from_checkpoint_also_handles_legacy_state_dict_key(tmp_path: Path) -> None:
    """NeilCNNWrapper.from_checkpoint must also handle the legacy 'state_dict' key."""
    fake_weights = {"weight": MagicMock()}
    torch_stub = _build_torch_stub({"state_dict": fake_weights})
    torch_stub.cuda.is_available = MagicMock(return_value=False)

    fake_model = MagicMock()
    import unittest.mock as mock

    with mock.patch.dict(
        sys.modules, {"torch": torch_stub, "torch.cuda": torch_stub.cuda}
    ):
        NeilCNNWrapper.from_checkpoint(
            tmp_path / "best.pt",
            model_builder=lambda: fake_model,
        )
    fake_model.load_state_dict.assert_called_once_with(fake_weights)


def test_score_candidate_pool_accepts_image_head_probabilities_dataclass() -> None:
    """score_candidate_pool must auto-convert ImageHeadProbabilities via .as_dict()."""
    probs_dataclass = ImageHeadProbabilities(
        formal=np.array([0.8, 0.1, 0.1], dtype=np.float32),
        season=np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
        time=np.array([0.9, 0.1], dtype=np.float32),
        gender=np.array([0.6, 0.3, 0.1], dtype=np.float32),
        frequency=np.array([0.3, 0.7], dtype=np.float32),
    )
    probs_dict = probs_dataclass.as_dict()
    candidates = [
        {"formality": 0.2, "day_night": 0.2, "likely_season": "spring", "gender": "male", "frequency": "everyday"},
        {"formality": 0.8, "day_night": 0.8, "likely_season": "winter", "gender": "female", "frequency": "occasional"},
    ]
    scores_via_dataclass = score_candidate_pool(probs_dataclass, candidates)
    scores_via_dict = score_candidate_pool(probs_dict, candidates)
    assert np.allclose(scores_via_dataclass, scores_via_dict)
