from hybrid_rag.fusion import rrf_fuse


def test_fuses_two_lists():
    list1 = [("a", 0.9), ("b", 0.5)]
    list2 = [("b", 0.8), ("c", 0.2)]
    fused = rrf_fuse([list1, list2], k=60)
    assert fused[0][0] == "b"
    assert fused[1][0] == "a"
    assert fused[2][0] == "c"


def test_doc_present_in_one_list_still_scores():
    list1 = [("x", 1.0)]
    list2 = [("y", 1.0), ("z", 0.5)]
    fused = rrf_fuse([list1, list2], k=60)
    ids = [d for d, _ in fused]
    assert "x" in ids and "y" in ids and "z" in ids


def test_weights_change_order():
    list1 = [("a", 1.0), ("b", 0.5)]
    list2 = [("b", 1.0), ("a", 0.5)]
    equal = rrf_fuse([list1, list2], weights=[1.0, 1.0])
    biased = rrf_fuse([list1, list2], weights=[10.0, 0.1])
    assert [d for d, _ in equal][0] == "a" or [d for d, _ in equal][0] == "b"
    assert [d for d, _ in biased][0] == "a"


def test_rank_score_reciprocal_decay():
    # Rank 1 contributes more than rank 2.
    single = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
    fused = rrf_fuse([single], k=60)
    scores = {d: s for d, s in fused}
    assert scores["a"] > scores["b"] > scores["c"]
