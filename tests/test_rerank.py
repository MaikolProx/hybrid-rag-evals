"""Tests for cross-encoder reranker."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestRerankerEnabled:
    def test_disabled_by_default(self):
        from hybrid_rag.rerank import reranker_enabled
        with patch.dict(os.environ, {}, clear=True):
            assert reranker_enabled() is False

    def test_enabled_with_env(self):
        from hybrid_rag.rerank import reranker_enabled
        with patch.dict(os.environ, {"HYBRID_RAG_RERANK": "1"}):
            assert reranker_enabled() is True

    def test_disabled_with_other_value(self):
        from hybrid_rag.rerank import reranker_enabled
        with patch.dict(os.environ, {"HYBRID_RAG_RERANK": "0"}):
            assert reranker_enabled() is False

    def test_disabled_with_empty_string(self):
        from hybrid_rag.rerank import reranker_enabled
        with patch.dict(os.environ, {"HYBRID_RAG_RERANK": ""}):
            assert reranker_enabled() is False


class TestCrossEncoderReranker:
    @patch("sentence_transformers.CrossEncoder")
    def test_init_stores_model(self, mock_ce):
        from hybrid_rag.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name="test-model")
        mock_ce.assert_called_once_with("test-model")
        assert reranker._docs == {}

    @patch("sentence_transformers.CrossEncoder")
    def test_set_docs(self, mock_ce):
        from hybrid_rag.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        docs = {"doc1": "content1", "doc2": "content2"}
        reranker.set_docs(docs)
        assert reranker._docs == docs

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_returns_top_k(self, mock_ce):
        from hybrid_rag.rerank import CrossEncoderReranker
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5, 0.3]
        mock_ce.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranker.set_docs({"d1": "text1", "d2": "text2", "d3": "text3"})
        candidates = [("d1", 0.8), ("d2", 0.7), ("d3", 0.6)]
        results = reranker.rerank("query", candidates, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "d1"  # highest score
        assert results[0][1] == 0.9

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_uses_docs_map(self, mock_ce):
        from hybrid_rag.rerank import CrossEncoderReranker
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        mock_ce.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranker.set_docs({"d1": "real content"})
        reranker.rerank("query", [("d1", 0.5)])

        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [("query", "real content")]

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_fallback_to_doc_id(self, mock_ce):
        from hybrid_rag.rerank import CrossEncoderReranker
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.7]
        mock_ce.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranker.rerank("query", [("unknown_doc", 0.5)])

        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [("query", "unknown_doc")]
