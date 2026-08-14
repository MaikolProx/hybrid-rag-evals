from hybrid_rag.eval import (
    dcg_at_k,
    evaluate_retrieval,
    lexical_faithfulness,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=1) == 0.5
    assert recall_at_k(["x", "y"], {"z"}, k=2) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 2 / 3
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_dcg_and_ndcg_known_values():
    # DCG@3 con relevantes en posiciones 1 y 3: 1 + 0.5 = 1.5
    assert abs(dcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) - 1.5) < 1e-6
    # IDCG con 2 relevantes: 1 + 0.6309 = 1.6309
    ideal = 1.0 + 1.0 / 1.5849625007211563
    assert abs(ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) - (1.5 / ideal)) < 1e-6
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_evaluate_retrieval_aggregates():
    ranked = {
        "q1": ["a", "b", "c", "d", "e"],   # relevant a en rank 1 -> RR 1.0
        "q2": ["x", "y", "e", "a", "b"],   # relevant e en rank 3 -> RR 1/3, NDCG 0.5
    }
    qrels = {"q1": {"a"}, "q2": {"e"}}
    res = evaluate_retrieval(ranked, qrels, k=5)
    assert res["recall@5"] == 1.0
    assert abs(res["mrr"] - (1.0 + 1 / 3) / 2) < 1e-4
    assert abs(res["ndcg@5"] - 0.75) < 1e-6  # (1.0 + 0.5)/2


def test_lexical_faithfulness_overlap():
    assert lexical_faithfulness("el perro ladra", "un perro ladra fuerte") > 0.5
    assert lexical_faithfulness("las ballenas nadan en el océano", "el modelo de lenguaje") == 0.0
