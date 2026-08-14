# Reporte de evaluación — Hybrid RAG

Generado: Fri 08/14/2026 | Corpus: 21 chunks | K=5 | Consultas: 14

| Sistema | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|
| BM25 (baseline) | 0.643 | 0.607 | 0.617 |
| Hybrid (BM25+dense+RRF) | 0.821 | 0.677 | 0.702 |
| Hybrid + rerank (cross-encoder) | 0.964 | 0.812 | 0.845 |

**Faithfulness** (muestra de 14 consultas): léxico 0.836 | LLM-as-judge 0.000 (sin key — fallback léxico)

## Mejora del híbrido sobre baseline
- `¿cómo impedir que la herramienta responda con información inventada?` -> BM25 0/2, Hybrid 1/2 + mejora
- `cuando algo falla a mitad el proceso continúa desde el paso anterior` -> BM25 0/1, Hybrid 1/1 + mejora
- `evitar llamadas repetidas para reducir la factura` -> BM25 0/1, Hybrid 1/1 + mejora
- `descubrir los servidores ocultos de una empresa` -> BM25 0/1, Hybrid 0/1 =
- `saber qué procesos escuchan en cada puerto de la red` -> BM25 0/1, Hybrid 0/1 =
- `BM25 k1 b frecuencia de términos en el corpus` -> BM25 1/1, Hybrid 1/1 =
- `Content-Security-Policy Strict-Transport-Security auditoría` -> BM25 1/1, Hybrid 1/1 =
- `pgvector índice aproximado millones de vectores` -> BM25 1/1, Hybrid 1/1 =
- `métricas recall MRR NDCG` -> BM25 1/1, Hybrid 1/1 =
- `combinar varias listas de resultados sin normalizar puntuaciones` -> BM25 1/1, Hybrid 1/1 =
- `fase amplia barata y luego refinar con un segundo modelo` -> BM25 1/1, Hybrid 1/1 =
- `órdenes ocultas dentro de un documento recuperado` -> BM25 2/2, Hybrid 2/2 =
- `monitorear coste latencia y deriva de calidad por llamada` -> BM25 1/1, Hybrid 1/1 =
- `pausar la ejecución y reanudar en el mismo punto del flujo` -> BM25 1/1, Hybrid 1/1 =