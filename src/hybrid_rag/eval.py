"""Evaluation metrics for retrieval and faithfulness.

All ranking metrics are implemented from scratch (numpy/scipy-free) so the
formulas are explicit and testable:

  - recall@k : |relevant ∩ top-k| / |relevant|
  - precision@k : |relevant ∩ top-k| / k
  - MRR       : mean over queries of 1/rank(first relevant hit)
  - NDCG@k    : DCG@k / IDCG@k with gain 1 at relevant positions

Faithfulness has two modes:
  - lexical fallback: token-overlap between the answer and its source chunk
    (Jaccard-style), useful offline as a lower bound.
  - LLM-as-judge: a 1-5 faithfulness score from an LLM over the source chunk,
    gated on the presence of an OPENAI_API_KEY.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

from .tokenizer import tokenize


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    top = ranked_ids[:k]
    return len(set(top) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    top = ranked_ids[:k]
    return len(set(top) & relevant) / k if k else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    # DCG = sum over relevant docs at position i of gain/log2(i+1), gain=1.
    return sum(
        1.0 / log2(i + 1)
        for i, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant
    )


def log2(x: float) -> float:
    """Explicit log2 to keep the module dependency-free."""
    import math

    return math.log2(x)


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    dcg = dcg_at_k(ranked_ids, relevant, k)
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean(items: Sequence[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def evaluate_retrieval(
    ranked_lists: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int = 5,
) -> dict[str, float]:
    """Compute recall@k, MRR and NDCG@k across the query set."""
    recalls, mrr_vals, ndcgs = [], [], []
    for query, relevant in qrels.items():
        ranked = ranked_lists.get(query, [])
        recalls.append(recall_at_k(ranked, relevant, k))
        mrr_vals.append(reciprocal_rank(ranked, relevant))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
    return {
        f"recall@{k}": round(mean(recalls), 4),
        "mrr": round(mean(mrr_vals), 4),
        f"ndcg@{k}": round(mean(ndcgs), 4),
    }


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

def lexical_faithfulness(answer: str, source: str) -> float:
    """Token-overlap between answer and source chunk (Jaccard-like).

    Lower bound signal: if even the tokens don't overlap, the answer is not
    grounded in the retrieved chunk.
    """
    a = set(tokenize(answer))
    s = set(tokenize(source))
    if not a or not s:
        return 0.0
    return len(a & s) / len(a)


def llm_judge_faithfulness(query: str, answer: str, source: str) -> float | None:
    """LLM-as-judge: faithfulness of `answer` to `source` on a 1-5 scale.

    Returns None when no OPENAI_API_KEY is configured.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    client = OpenAI()
    prompt = (
        "Evalúa la fidelidad (faithfulness) de la respuesta respecto a la "
        "fuente dada. Respuesta: solo 1 número entero 1-5 donde 5 = cada "
        "afirmación está respaldada por la fuente, 1 = inventa hechos.\n\n"
        f"FUENTE:\n{source[:1500]}\n\nRESPUESTA:\n{answer[:500]}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content
        if raw is None:
            return None
        score = int(raw.strip())
        return max(1, min(5, score))
    except Exception:
        return None
