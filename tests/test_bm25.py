from hybrid_rag.bm25 import BM25
from hybrid_rag.tokenizer import tokenize

CORPUS = [
    ("a", "el perro ladra fuerte en el parque cada mañana"),
    ("b", "el gato duerme sobre el sofá de la sala"),
    ("c", "los perros y los gatos son mascotas populares en las casas"),
    ("d", "los coches eléctricos se cargan en estaciones públicas"),
    ("e", "los coches de gasolina consumen mucho combustible"),
]


def test_relevant_doc_ranks_first():
    bm25 = BM25(CORPUS, tokenizer=tokenize)
    hits = bm25.search("perro ladrando en el parque", top_k=3)
    assert hits[0][0] == "a"


def test_semantic_neighbor_via_shared_terms():
    bm25 = BM25(CORPUS, tokenizer=tokenize)
    hits = bm25.search("coches eléctricos", top_k=2)
    assert hits[0][0] == "d"


def test_scores_decrease_monotonically():
    bm25 = BM25(CORPUS, tokenizer=tokenize)
    hits = bm25.search("coches", top_k=5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_unknown_term_returns_nothing():
    bm25 = BM25(CORPUS, tokenizer=tokenize)
    hits = bm25.search("xyzq", top_k=5)
    assert hits == []


def test_idf_weights_rare_terms():
    # "estaciones" only appears in doc d -> searching for it must rank d top.
    bm25 = BM25(CORPUS, tokenizer=tokenize)
    hits = bm25.search("estaciones públicas", top_k=3)
    assert hits[0][0] == "d"


def test_top_k_is_respected():
    bm25 = BM25(CORPUS, tokenizer=tokenize)
    hits = bm25.search("coches", top_k=2)
    assert len(hits) == 2
