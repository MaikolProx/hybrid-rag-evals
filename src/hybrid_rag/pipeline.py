"""End-to-end hybrid retrieval pipeline.

Builds a BM25 index and a dense index over the same chunked documents,
retrieves candidates from both, fuses them with RRF, and optionally reranks.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .bm25 import BM25
from .embeddings import DenseIndex
from .fusion import rrf_fuse
from .rerank import CrossEncoderReranker, reranker_enabled


class HybridRetriever:
    def __init__(
        self,
        dense_index: Optional[DenseIndex] = None,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        rerank_top_n: int = 20,
    ) -> None:
        self.dense = dense_index or DenseIndex()
        self.bm25: Optional[BM25] = None
        self.rrf_k = rrf_k
        self.weights = [bm25_weight, dense_weight]
        self.rerank_top_n = rerank_top_n
        self.doc_map: Dict[str, str] = {}
        self._reranker: Optional[CrossEncoderReranker] = None

    def build(self, docs: List[tuple[str, str]]) -> None:
        """docs: [(doc_id, text), ...]"""
        self.doc_map = dict(docs)
        self.bm25 = BM25(docs, k1=self.bm25.k1 if self.bm25 else 1.5, b=self.bm25.b if self.bm25 else 0.75)
        self.dense.add([d for d, _ in docs], [t for _, t in docs])
        if reranker_enabled():
            self._reranker = CrossEncoderReranker()
            self._reranker.set_docs(self.doc_map)

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("build() must be called before retrieve()")
        bm25_hits = self.bm25.search(query, top_k=self.rerank_top_n)
        dense_hits = self.dense.search(query, top_k=self.rerank_top_n)
        fused = rrf_fuse([bm25_hits, dense_hits], k=self.rrf_k, weights=self.weights)
        if self._reranker is not None:
            return self._reranker.rerank(query, fused[: self.rerank_top_n], top_k=top_k)
        return fused[:top_k]

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None
