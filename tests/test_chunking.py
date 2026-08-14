from hybrid_rag.chunking import chunk_markdown


def test_splits_by_headings():
    md = "# Título A\n\ncontenido uno\n\n## Subtítulo\n\ndetalle\n\n# Título B\n\ncontenido dos\n"
    chunks = chunk_markdown(md, max_chars=10000)
    assert len(chunks) == 2
    assert "contenido uno" in chunks[0][1]
    assert "contenido dos" in chunks[1][1]


def test_long_section_split_by_paragraphs():
    body = "\n\n".join(f"párrafo {i} " + "x" * 100 for i in range(10))
    md = f"# Encabezado\n\n{body}\n"
    chunks = chunk_markdown(md, max_chars=300)
    texts = [c for _, c in chunks]
    assert len(texts) > 1
    assert all(len(c) <= 500 for c in texts)


def test_empty_body_ignored():
    md = "# Solo\n\n## Vacío\n"
    chunks = chunk_markdown(md)
    assert chunks == []
