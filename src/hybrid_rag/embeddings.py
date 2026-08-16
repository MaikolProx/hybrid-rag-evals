"""Dense embedding index on top of sentence-transformers.

The model is loaded lazily (first call) so that CLI help / tests that never
hit the vector path stay fast and offline-friendly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class DenseIndex:
    """Cosine-similarity vector store over an embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None
        self.doc_ids: list[str] = []
        self._vectors: NDArray[np.float32] | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        model = self._get_model()
        result = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(result, dtype=np.float32)

    def add(self, doc_ids: list[str], texts: list[str]) -> None:
        vectors = self.embed(texts)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self.doc_ids.extend(doc_ids)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._vectors is None or len(self.doc_ids) == 0:
            return []
        q = self.embed([query])[0]
        # Both normalized -> dot product == cosine similarity.
        sims = self._vectors @ q
        order = np.argsort(-sims, kind="stable")[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in order]
