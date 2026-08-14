"""BM25 (Okapi) ranking from scratch, numpy-only.

Reference: Robertson & Zaragoza, "The Probabilistic Relevance Framework:
BM25 and Beyond" (2009).

Scores are computed over a pre-built inverted index:
  - idf uses the standard Robertson-Sparck Jones form with smoothing:
      idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
  - term frequency component saturates via k1 and length normalization
    via b over the average document length.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np

from .tokenizer import tokenize


class BM25:
    """Okapi BM25 ranker.

    The corpus is supplied as a list of (doc_id, text) pairs. The tokenizer
    used at query time must match the one used at index time.
    """

    def __init__(
        self,
        corpus: Sequence[tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: callable = tokenize,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.doc_ids: List[str] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self._df: Dict[str, int] = defaultdict(int)
        self._postings: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc_id: tf}
        self._build(corpus)

    def _build(self, corpus: Sequence[tuple[str, str]]) -> None:
        doc_term_counts: Dict[str, Counter] = {}
        for doc_id, text in corpus:
            self.doc_ids.append(doc_id)
            tokens = self.tokenizer(text)
            self.doc_lens.append(len(tokens))
            doc_term_counts[doc_id] = Counter(tokens)
        if self.doc_lens:
            self.avg_doc_len = float(np.mean(self.doc_lens))
        for doc_id, counts in doc_term_counts.items():
            for term, tf in counts.items():
                self._postings[term][doc_id] = tf
        for term in self._postings:
            self._df[term] = len(self._postings[term])

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _score_doc(self, term: str, doc_id: str, dl: int) -> float:
        tf = self._postings[term].get(doc_id, 0)
        if not tf:
            return 0.0
        idf = self._idf(term)
        denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len)
        return idf * (tf * (self.k1 + 1.0)) / denom

    def search(self, query: str, top_k: int = 10) -> List[tuple[str, float]]:
        """Return [(doc_id, score), ...] sorted by descending score.

        Documents that do not contain any query term get score 0 and are
        included at the end with score 0.0 only if fewer than top_k docs match.
        """
        terms = self.tokenizer(query)
        if not terms:
            return []
        scores = np.zeros(len(self.doc_ids))
        for term in set(terms):
            for doc_id in self._postings.get(term, {}):
                idx = self.doc_ids.index(doc_id)
                scores[idx] += self._score_doc(term, doc_id, self.doc_lens[idx])
        order = np.argsort(-scores, kind="stable")
        results = [(self.doc_ids[i], float(scores[i])) for i in order]
        # Keep matched docs first, drop zero-score tail beyond matched count.
        matched = [(d, s) for d, s in results if s > 0.0]
        return matched[:top_k] if len(matched) >= top_k else matched
