from hybrid_rag.tokenizer import normalize, tokenize


def test_tokenize_removes_stopwords():
    tokens = tokenize("el perro corre rápido y salta", stopwords=True)
    assert "el" not in tokens
    assert "y" not in tokens
    assert "perro" in tokens
    assert "corre" in tokens


def test_tokenize_keeps_unicode_accents():
    tokens = tokenize("máquina de búsqueda híbrida", stopwords=True)
    assert "máquina" in tokens
    assert "híbrida" in tokens


def test_tokenize_numbers_and_short_tokens():
    tokens = tokenize("RAG v2 y BM25", stopwords=True)
    assert "rag" in tokens
    assert "bm25" in tokens
    assert not any(len(t) < 2 for t in tokens)


def test_normalize_lowercases():
    assert normalize("Hola Mundo") == "hola mundo"
