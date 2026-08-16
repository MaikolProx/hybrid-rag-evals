"""Tests for HybridRetriever pipeline (with mocked dense index)."""
from __future__ import annotations

from unittest.mock import MagicMock

from hybrid_rag.pipeline import HybridRetriever


class MockDenseIndex:
    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        self._vectors = None

    def add(self, doc_ids: list[str], texts: list[str]) -> None:
        self.doc_ids = doc_ids

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return [(doc_id, 0.5) for doc_id in self.doc_ids[:top_k]]


def test_hybrid_retriever_build_and_retrieve():
    docs = [
        ("doc1", "This is the first document about Python programming."),
        ("doc2", "Second document about machine learning and AI."),
        ("doc3", "Third document about web development with JavaScript."),
    ]
    retriever = HybridRetriever(dense_index=MockDenseIndex())
    retriever.build(docs)

    assert retriever.bm25 is not None
    assert len(retriever.doc_map) == 3

    results = retriever.retrieve("Python", top_k=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_hybrid_retriever_requires_build():
    retriever = HybridRetriever(dense_index=MockDenseIndex())
    try:
        retriever.retrieve("test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "build() must be called" in str(e)


def test_hybrid_retriever_has_reranker_false_by_default():
    retriever = HybridRetriever(dense_index=MockDenseIndex())
    retriever.build([("doc1", "test")])
    assert retriever.has_reranker is False