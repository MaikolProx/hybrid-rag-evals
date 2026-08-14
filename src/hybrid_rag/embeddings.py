"""Dense embedding index on top of sentence-transformers.

The model is loaded lazily (first call) so that CLI help / tests that never
hit the vector path stay fast and offline-friendly.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class DenseIndex:
    """Cosine-similarity vector store over an embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self.doc_ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def add(self, doc_ids: List[str], texts: List[str]) -> None:
        vectors = self.embed(texts)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self.doc_ids.extend(doc_ids)

    def search(self, query: str, top_k: int = 10) -> List[tuple[str, float]]:
        if self._vectors is None or len(self.doc_ids) == 0:
            return []
        q = self.embed([query])[0]
        # Both normalized -> dot product == cosine similarity.
        sims = self._vectors @ q
        order = np.argsort(-sims, kind="stable")[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in order]
