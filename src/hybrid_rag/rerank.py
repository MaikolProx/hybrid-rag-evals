"""Optional cross-encoder reranking.

Uses a cross-encoder (ms-marco-MiniLM-L-6-v2) to score (query, doc) pairs and
re-order the candidates. To bound cost, rerank only runs over the already
fused top-N candidates.

Disabled by default to keep the demo fully offline. Enable with:

    HYBRID_RAG_RERANK=1 python scripts/run_evals.py
"""
from __future__ import annotations

import os
from typing import Dict, List


class CrossEncoderReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self._docs: Dict[str, str] = {}

    def set_docs(self, doc_map: Dict[str, str]) -> None:
        self._docs = doc_map

    def rerank(self, query: str, candidates: List[tuple[str, float]], top_k: int = 5) -> List[tuple[str, float]]:
        """Score (query, doc) pairs and return the top-k as [(doc_id, score)]."""
        pairs = [(query, self._docs.get(doc_id, doc_id)) for doc_id, _ in candidates]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(candidates, scores), key=lambda item: -float(item[1]))
        return [(doc_id, float(score)) for (doc_id, _), score in ranked[:top_k]]


def reranker_enabled() -> bool:
    return os.environ.get("HYBRID_RAG_RERANK", "0") == "1"
