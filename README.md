# hybrid-rag-evals

Retrieval híbrido para RAG: BM25 + embeddings + fusión RRF + reranker opcional, con una suite de evals que mide cada cambio.

Lo escribí desde cero con numpy puro por dos razones. Una: quiero ver qué hace cada pieza y poder explicarla en una entrevista. Otra: cuando algo falla, la falla es del algoritmo que entiendo, no de una librería que no entiendo.

## Números

Corpus de demo: 21 documentos con distractores léxicos y 14 consultas etiquetadas a mano. Reporte completo en `evals/report.md`.

| Sistema | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|
| BM25 (baseline) | 0.643 | 0.607 | 0.617 |
| Híbrido (BM25 + dense + RRF) | 0.821 | 0.677 | 0.702 |
| Híbrido + rerank (cross-encoder) | 0.964 | 0.812 | 0.845 |

Lo que la tabla dice en términos simples: 5 de las 14 consultas no comparten ninguna palabra con el documento correcto. BM25 saca cero en esas cinco. El híbrido rescata tres y el reranker llega a las cinco.

También dice dónde falla. El híbrido sin reranker pierde dos consultas semánticas. Saberlo antes de desplegar es el punto de tener evals. Si mañana un cambio baja el recall, el reporte lo muestra.

## Cómo está hecho

- Chunking por encabezados (`#`, `##`, `###`) con límite de caracteres.
- BM25 con k1=1.5, b=0.75, tokenizador en español con stopwords.
- Denso con `all-MiniLM-L6-v2` (384 dims), coseno en numpy.
- RRF con k=60. No normalizo scores entre retrievers, uso rangos.
- Rerank opcional con `ms-marco-MiniLM-L-6-v2` sobre el top-20 fusionado.
- Métricas de ranking escritas a mano y faithfulness (léxico, o LLM-as-judge si hay `OPENAI_API_KEY`).

## Para correrlo

```bash
pip install -r requirements.txt
python scripts/run_evals.py
pytest -q
```

La única pieza que necesita red y una key es el LLM-as-judge. Sin key, usa el fallback léxico y no falla.

## Decisiones que me llevaron aquí

- BM25 propio en vez de `rank_bm25`: menos dependencias y comportamiento documentado.
- RRF en vez de sumar scores normalizados: un score de BM25 y una similitud coseno no son comparables.
- Rerank solo sobre el top-20 fusionado: el cross-encoder es caro por par.
- Todo corre local. Reproducible en cualquier máquina con CPU.

## Estructura

```
src/hybrid_rag/
  tokenizer.py     tokenización + stopwords (es)
  bm25.py          scoring BM25 desde cero
  embeddings.py    wrapper sentence-transformers (carga perezosa)
  vector_store.py  índice denso + búsqueda coseno
  fusion.py        RRF con ponderación
  rerank.py        cross-encoder opcional
  chunking.py      chunking por encabezados
  pipeline.py      retrieval híbrido end-to-end
  eval.py          recall@k / precision@k / MRR / NDCG@k / faithfulness
data/corpus.md
scripts/run_evals.py
tests/
evals/report.md
```
