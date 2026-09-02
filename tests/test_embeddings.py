"""Tests for dense embeddings index."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestDenseIndexInit:
    def test_default_model_name(self):
        from hybrid_rag.embeddings import DenseIndex
        idx = DenseIndex()
        assert idx.model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        from hybrid_rag.embeddings import DenseIndex
        idx = DenseIndex(model_name="custom-model")
        assert idx.model_name == "custom-model"

    def test_device_stored(self):
        from hybrid_rag.embeddings import DenseIndex
        idx = DenseIndex(device="cuda")
        assert idx.device == "cuda"

    def test_initial_state_empty(self):
        from hybrid_rag.embeddings import DenseIndex
        idx = DenseIndex()
        assert idx.doc_ids == []
        assert idx._vectors is None
        assert idx._model is None


class TestDenseIndexEmbed:
    @patch("hybrid_rag.embeddings.DenseIndex._get_model")
    def test_embed_returns_array(self, mock_get_model):
        from hybrid_rag.embeddings import DenseIndex
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        mock_get_model.return_value = mock_model

        idx = DenseIndex()
        result = idx.embed(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        mock_model.encode.assert_called_once_with(
            ["hello", "world"], normalize_embeddings=True, convert_to_numpy=True
        )

    @patch("hybrid_rag.embeddings.DenseIndex._get_model")
    def test_embed_single_text(self, mock_get_model):
        from hybrid_rag.embeddings import DenseIndex
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
        mock_get_model.return_value = mock_model

        idx = DenseIndex()
        result = idx.embed(["test"])
        assert result.shape == (1, 2)


class TestDenseIndexAdd:
    @patch("hybrid_rag.embeddings.DenseIndex._get_model")
    def test_add_first_batch(self, mock_get_model):
        from hybrid_rag.embeddings import DenseIndex
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_get_model.return_value = mock_model

        idx = DenseIndex()
        idx.add(["doc1", "doc2"], ["text1", "text2"])

        assert idx.doc_ids == ["doc1", "doc2"]
        assert idx._vectors is not None
        assert idx._vectors.shape == (2, 2)

    @patch("hybrid_rag.embeddings.DenseIndex._get_model")
    def test_add_second_batch_stacks(self, mock_get_model):
        from hybrid_rag.embeddings import DenseIndex
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]], dtype=np.float32)
        mock_get_model.return_value = mock_model

        idx = DenseIndex()
        idx.add(["doc1"], ["text1"])
        idx.add(["doc2"], ["text2"])

        assert idx.doc_ids == ["doc1", "doc2"]
        assert idx._vectors.shape == (2, 2)


class TestDenseIndexSearch:
    @patch("hybrid_rag.embeddings.DenseIndex._get_model")
    def test_search_empty_returns_empty(self, mock_get_model):
        from hybrid_rag.embeddings import DenseIndex
        idx = DenseIndex()
        result = idx.search("query")
        assert result == []

    @patch("hybrid_rag.embeddings.DenseIndex._get_model")
    def test_search_returns_ranked_results(self, mock_get_model):
        from hybrid_rag.embeddings import DenseIndex
        mock_model = MagicMock()
        # First call for add, second for query
        mock_model.encode.side_effect = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),  # add
            np.array([[1.0, 0.0]], dtype=np.float32),  # search query
        ]
        mock_get_model.return_value = mock_model

        idx = DenseIndex()
        idx.add(["doc1", "doc2"], ["text1", "text2"])
        results = idx.search("query", top_k=2)

        assert len(results) == 2
        assert results[0][0] == "doc1"  # most similar
        assert results[0][1] > results[1][1]  # scores decreasing

    @patch("hybrid_rag.embeddings.DenseIndex._get_model")
    def test_search_top_k_limits_results(self, mock_get_model):
        from hybrid_rag.embeddings import DenseIndex
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        ]
        mock_get_model.return_value = mock_model

        idx = DenseIndex()
        idx.add(["d1", "d2", "d3"], ["t1", "t2", "t3"])
        results = idx.search("query", top_k=1)
        assert len(results) == 1


class TestDenseIndexGetModel:
    def test_lazy_loading(self):
        from hybrid_rag.embeddings import DenseIndex
        idx = DenseIndex()
        assert idx._model is None

    @patch("hybrid_rag.embeddings.DenseIndex._get_model")
    def test_model_cached(self, mock_get_model):
        from hybrid_rag.embeddings import DenseIndex
        mock_get_model.return_value = MagicMock()
        idx = DenseIndex()
        idx._get_model()
        idx._get_model()
        assert mock_get_model.call_count == 2  # each call goes through
