from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from vibescents.reranker import Qwen3VLReranker, GeminiReranker
from vibescents.schemas import RerankResponse
from vibescents.settings import Settings


def _build_ml_stubs():
    """Builds stubs for torch, transformers, and qwen_vl_utils to avoid heavy imports."""
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.no_grad = MagicMock()
    torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)
    torch.tensor = MagicMock()
    torch.softmax = MagicMock()

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = MagicMock()
    transformers.AutoModelForCausalLM = MagicMock()

    qwen_vl_utils = types.ModuleType("qwen_vl_utils")
    qwen_vl_utils.process_vision_info = MagicMock(return_value=([], None))

    return {
        "torch": torch,
        "transformers": transformers,
        "qwen_vl_utils": qwen_vl_utils,
    }


def test_reranker_init_loads_model() -> None:
    """Assert Qwen3VLReranker.__init__ completes and loads model/processor."""
    stubs = _build_ml_stubs()
    mock_proc = MagicMock()
    mock_proc.tokenizer.encode.side_effect = lambda x, **kwargs: (
        [1] if x == "yes" else [2]
    )
    stubs["transformers"].AutoProcessor.from_pretrained.return_value = mock_proc

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    stubs["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model

    with patch.dict(sys.modules, stubs):
        settings = Settings(reranker_model="mock-model")
        reranker = Qwen3VLReranker(settings=settings)

        assert reranker._yes_id == 1
        assert reranker._no_id == 2
        stubs["transformers"].AutoProcessor.from_pretrained.assert_called_once()
        stubs["transformers"].AutoModelForCausalLM.from_pretrained.assert_called_once()


def test_rerank_returns_sorted_results() -> None:
    """Assert rerank returns results sorted by score descending."""
    stubs = _build_ml_stubs()
    torch = stubs["torch"]

    with patch.dict(sys.modules, stubs):
        # Manually construct to avoid __init__ overhead
        reranker = Qwen3VLReranker.__new__(Qwen3VLReranker)
        reranker._yes_id = 1
        reranker._no_id = 2
        reranker._proc = MagicMock()
        reranker._model = MagicMock()

        # Mock candidates
        class Candidate:
            def __init__(self, fid, text):
                self.fragrance_id = fid
                self.retrieval_text = text

        candidates = [
            Candidate(fid="bad", text="Smells like burning rubber"),
            Candidate(fid="good", text="Divine jasmine and rose"),
        ]

        # Mock model output: pointwise calls.
        mock_logits = MagicMock()

        def get_item(idx):
            # idx is _yes_id or _no_id
            if idx == 1:  # yes
                return MagicMock(
                    item=lambda: 0.0 if reranker._model.call_count == 1 else 10.0
                )
            return MagicMock(
                item=lambda: 10.0 if reranker._model.call_count == 1 else 0.0
            )

        mock_logits.__getitem__.side_effect = get_item
        reranker._model.return_value.logits.__getitem__.return_value = mock_logits

        # Mock softmax to return [0.1] then [0.9]
        torch.softmax.side_effect = [
            [0.1],  # score for 'bad'
            [0.9],  # score for 'good'
        ]

        response = reranker.rerank(
            occasion_text="Summer Wedding", candidates=candidates
        )

        assert isinstance(response, RerankResponse)
        assert len(response.results) == 2
        # Sorted descending: good (0.9) first, bad (0.1) second
        assert response.results[0].fragrance_id == "good"
        assert response.results[0].overall_score == 0.9
        assert response.results[1].fragrance_id == "bad"
        assert response.results[1].overall_score == 0.1


def test_rerank_without_image_path(tmp_path) -> None:
    """Assert rerank works without image_path and skips image parts."""
    stubs = _build_ml_stubs()
    torch = stubs["torch"]

    with patch.dict(sys.modules, stubs):
        reranker = Qwen3VLReranker.__new__(Qwen3VLReranker)
        reranker._yes_id = 1
        reranker._no_id = 2
        reranker._proc = MagicMock()
        reranker._model = MagicMock()

        mock_logits = MagicMock()
        mock_logits.__getitem__.return_value = MagicMock(item=lambda: 5.0)
        reranker._model.return_value.logits.__getitem__.return_value = mock_logits
        torch.softmax.return_value = [0.5]

        class Candidate:
            fragrance_id = "1"
            retrieval_text = "test"

        reranker.rerank(occasion_text="test", candidates=[Candidate()], image_path=None)

        # Verify apply_chat_template was called with user content that has NO image
        messages = reranker._proc.apply_chat_template.call_args[0][0]
        user_content = messages[1]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"


def test_gemini_reranker_alias() -> None:
    """Assert GeminiReranker is an alias for Qwen3VLReranker."""
    assert GeminiReranker is Qwen3VLReranker
