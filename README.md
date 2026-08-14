# Hybrid RAG with Retrieval Evals

> Producción RAG que combina búsqueda **léxica (BM25) + densa (embeddings)**, fusiona con **RRF**, opcionalmente **rerank** con cross-encoder, y — lo más importante — **mide su calidad** con una suite de evals reproducible (recall@k, MRR, NDCG@5, faithfulness).

No es un wrapper de una librería: el BM25, la fusión RRF, el scoring de similitud y las métricas de evaluación están **implementados desde cero** (numpy puro) para que el comportamiento sea transparente y auditable.

---

## ¿Por qué existe este proyecto?

En producción, un RAG con solo embeddings falla en consultas con términos exactos (acrónimos, IDs, nombres propios) y un RAG con solo BM25 falla en consultas semánticas. La combinación híbrida mejora recall mediblemente, pero la mayoría de los equipos **no mide el impacto** porque montar un harness de evals da trabajo.

Este repo resuelve ambas cosas: implementa el retrieval híbrido **y** el harness que lo evalúa. Cualquier cambio (nuevo chunker, otro embedding, más rerankers) se valida contra la misma suite con números before/after.

## Resultados (ver `evals/report.md`)

| Sistema | Recall@5 | MRR | NDCG@5 | Faithfulness* |
|---|---|---|---|---|
| BM25 (baseline) | 0.643 | 0.607 | 0.617 | 0.836 |
| Hybrid (BM25 + dense + RRF) | 0.821 | 0.677 | 0.702 | 0.836 |
| **Hybrid + rerank (cross-encoder)** | **0.964** | **0.812** | **0.845** | 0.836 |

\* faithfulness calculado con LLM-as-judge si `OPENAI_API_KEY` está configurada; si no, con solapamiento léxico.

**Lectura honesta de los números** (corpus demo de 21 docs, 14 consultas etiquetadas a mano):

- 5 de las 14 consultas son **semánticas con cero solapamiento léxico**: BM25 no encuentra nada (recall 0), el híbrido rescata 3/5 y el reranker llega a 5/5.
- La ganancia no es magia: es exactamente el caso de uso real donde los términos exactos fallan (sinónimos, jerga, preguntas parafraseadas) y donde un cross-encoder reordena los candidatos por significado.
- Las 2 consultas que fallaban a nivel híbrido quedaron **expuestas por el harness** antes de añadir el reranker — ese es el valor de medir: el reporte te dice dónde estás fallando antes de desplegar.

## Arquitectura

```
                    ┌─────────────┐
 query ───────────▶ │  BM25 (lex) │──┐
                    └─────────────┘  │
                    ┌─────────────┐  │  ┌───────────────┐   ┌──────────┐
 query ───────────▶ │  dense      │──┴─▶│ RRF (k=60)   │──▶│ rerank   │──▶ top-k
                    └─────────────┘     └───────────────┘   │ (opc.)   │
                                                   ┌───────┴──────────┘
                                                   ▼
                                           eval harness: recall@k / MRR /
                                           NDCG@k / faithfulness
```

- **Chunking**: secciones por encabezado (`#`, `##`, `###`) con límite de caracteres.
- **BM25**: implementación propia (k1=1.5, b=0.75), tokenizador en español con stopwords.
- **Denso**: `sentence-transformers` `all-MiniLM-L6-v2` (384 dims), similitud coseno con numpy.
- **Fusión**: Reciprocal Rank Fusion (RRF), k=60, ponderado.
- **Rerank**: opcional `ms-marco-MiniLM-L-6-v2` (cross-encoder), solo top-20 fusionado para limitar coste.
- **Evals**: métricas de ranking implementadas a mano + faithfulness (léxico o LLM-as-judge).

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_evals.py
```

Si tienes `OPENAI_API_KEY` en el entorno, el harness añadirá la puntuación LLM-as-judge de faithfulness. Sin la key, usa el fallback léxico (no falla).

## Tests

```bash
pytest -q
```

Cubren el tokenizador, el scoring BM25 (orden correcto y monotonía), el RRF (estabilidad ante listas incompletas), y las métricas de eval (recall/MRR/NDCG con ground truths conocidos).

## Estructura

```
src/hybrid_rag/
  tokenizer.py     # tokenización + stopwords (es)
  bm25.py          # scoring BM25 desde cero
  embeddings.py    # wrapper sentence-transformers (lazy load)
  vector_store.py  # índice denso + búsqueda coseno
  fusion.py        # RRF con ponderación
  rerank.py        # cross-encoder opcional
  chunking.py      # chunking por encabezados
  pipeline.py      # retrieval híbrido end-to-end
  eval.py          # recall@k / precision@k / MRR / NDCG@k / faithfulness
data/corpus.md     # corpus de demostración (público, sin secretos)
scripts/run_evals.py
tests/
evals/report.md    # reporte generado
```

## Decisiones y ADRs

- **BM25 propio vs `rank_bm25`**: una dependencia menos, y el comportamiento queda documentado y testeado (útil para entrevistas técnicas).
- **RRF vs linear score fusion**: RRF no necesita normalizar scores de retrievers heterogéneos; con `k=60` los rankings cortos no pierden valor.
- **Rerank sobre top-20 fusionado**: el cross-encoder es O(n²) por par; restringirlo a candidatos ya fusionados limita latencia a la vez que corrige el orden final.
- **Modelo local, sin API**: el RAG completo corre offline (solo la opción LLM-as-judge necesita key). Reproducible y barato de mantener.
