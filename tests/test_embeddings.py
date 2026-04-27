from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We import the classes inside the patch context in each test to ensure
# lazy imports in __init__ get the mocked modules.


def test_sentence_transformer_embed_documents_returns_float32_array() -> None:
    """Embed 3 texts, assert shape and dtype."""
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model

    # Mock encode to return a (3, 768) float32 array
    mock_model.encode.return_value = np.zeros((3, 768), dtype=np.float32)

    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        from vibescents.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_multimodal_documents(["text1", "text2", "text3"])

        assert result.shape == (3, 768)
        assert result.dtype == np.float32
        mock_model.encode.assert_called_once()


def test_nomic_model_prepends_doc_prefix() -> None:
    """nomic-ai/nomic-embed-text-v1.5 model prepends 'search_document: ' to each text."""
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    mock_model.encode.return_value = np.zeros((1, 768), dtype=np.float32)

    model_name = "nomic-ai/nomic-embed-text-v1.5"
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        from vibescents.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(model_name=model_name)
        embedder.embed_multimodal_documents(["hello"])

        # Verify the prefix was prepended
        expected_texts = ["search_document: hello"]
        # The first argument to encode is text_list
        args, _ = mock_model.encode.call_args
        assert args[0] == expected_texts


def test_non_nomic_model_no_prefix() -> None:
    """Other model names don't prepend prefix."""
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    mock_model.encode.return_value = np.zeros((1, 768), dtype=np.float32)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        from vibescents.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(model_name=model_name)
        embedder.embed_multimodal_documents(["hello"])

        args, _ = mock_model.encode.call_args
        assert args[0] == ["hello"]


def test_embed_documents_empty_list_returns_empty_array() -> None:
    """Empty list returns shape[0] == 0."""
    mock_st = MagicMock()
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        from vibescents.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_multimodal_documents([])

        assert result.shape[0] == 0


def test_embed_query_raises_not_implemented() -> None:
    """SentenceTransformerEmbedder.embed_multimodal_query raises NotImplementedError."""
    mock_st = MagicMock()
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        from vibescents.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        with pytest.raises(
            NotImplementedError, match="SentenceTransformerEmbedder is text-only"
        ):
            embedder.embed_multimodal_query(text="hello")


def test_qwen3vl_embed_query_text_only() -> None:
    """No image_path, item passed to inner.process() has only 'text' key."""
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"
    mock_qwen = MagicMock()
    mock_inner = MagicMock()
    mock_qwen.Qwen3VLEmbedder.return_value = mock_inner

    tensor_mock = MagicMock()
    tensor_mock.cpu.return_value.float.return_value.numpy.return_value = np.zeros(768)
    mock_inner.process.return_value = tensor_mock

    with patch.dict(
        sys.modules, {"torch": mock_torch, "vibescents.qwen3_vl_embedding": mock_qwen}
    ):
        from vibescents.embeddings import Qwen3VLMultimodalEmbedder

        embedder = Qwen3VLMultimodalEmbedder()
        embedder.embed_multimodal_query(text="stylish outfit")

        mock_inner.process.assert_called_once_with([{"text": "stylish outfit"}])


def test_qwen3vl_embed_query_with_image_path(tmp_path: Path) -> None:
    """With tmp_path image file, item has 'image' key."""
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"
    mock_qwen = MagicMock()
    mock_inner = MagicMock()
    mock_qwen.Qwen3VLEmbedder.return_value = mock_inner

    tensor_mock = MagicMock()
    tensor_mock.cpu.return_value.float.return_value.numpy.return_value = np.zeros(768)
    mock_inner.process.return_value = tensor_mock

    img_file = tmp_path / "outfit.jpg"
    img_file.write_text("fake image data")

    with patch.dict(
        sys.modules, {"torch": mock_torch, "vibescents.qwen3_vl_embedding": mock_qwen}
    ):
        from vibescents.embeddings import Qwen3VLMultimodalEmbedder

        embedder = Qwen3VLMultimodalEmbedder()
        embedder.embed_multimodal_query(text="stylish outfit", image_path=img_file)

        args, _ = mock_inner.process.call_args
        item = args[0][0]
        assert item["text"] == "stylish outfit"
        assert "image" in item
        assert item["image"] == str(img_file.resolve())


def test_qwen3vl_embed_documents_empty_returns_empty() -> None:
    """Empty list returns shape[0]==0 without calling inner.process()."""
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"
    mock_qwen = MagicMock()
    mock_inner = MagicMock()
    mock_qwen.Qwen3VLEmbedder.return_value = mock_inner

    # Mock tqdm.auto to avoid issues if not installed/available
    with patch.dict(
        sys.modules,
        {
            "torch": mock_torch,
            "vibescents.qwen3_vl_embedding": mock_qwen,
            "tqdm.auto": MagicMock(),
        },
    ):
        from vibescents.embeddings import Qwen3VLMultimodalEmbedder

        embedder = Qwen3VLMultimodalEmbedder()
        result = embedder.embed_multimodal_documents([])

        assert result.shape[0] == 0
        mock_inner.process.assert_not_called()
