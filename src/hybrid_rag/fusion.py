"""Reciprocal Rank Fusion (RRF).

Combines ranked lists from heterogeneous retrievers without needing to
normalize their raw scores. For each document the fused score is:

    score(d) = sum_r weight_r / (k + rank_r(d))

where k=60 by default (Cormack, Clarke & Buettcher, 2009). Documents absent
from a ranked list simply contribute 0 for that list.
"""
from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(
    ranked_lists: Sequence[list[tuple[str, float]]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several [(doc_id, score), ...] lists into one ranked list.

    Args:
        ranked_lists: each list is already sorted by descending relevance.
        k: RRF constant (default 60).
        weights: per-list weight; defaults to equal weighting.

    Returns:
        List of (doc_id, fused_score) sorted by descending score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])
