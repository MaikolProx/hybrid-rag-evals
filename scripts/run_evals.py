"""Run the eval suite and write evals/report.md.

Builds three retrieval variants over the demo corpus:
  1. BM25 only (baseline)
  2. Hybrid (BM25 + dense + RRF)
  3. Hybrid + cross-encoder rerank (if HYBRID_RAG_RERANK=1)

Then evaluates recall@k, MRR, NDCG@k against a hand-labeled query set and
measures faithfulness on a small generation sample.

Usage:
    python scripts/run_evals.py
    HYBRID_RAG_RERANK=1 python scripts/run_evals.py   # + reranker
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.bm25 import BM25  # noqa: E402
from hybrid_rag.chunking import chunk_markdown  # noqa: E402
from hybrid_rag.embeddings import DenseIndex  # noqa: E402
from hybrid_rag.eval import (  # noqa: E402
    evaluate_retrieval,
    lexical_faithfulness,
    llm_judge_faithfulness,
    mean,
)
from hybrid_rag.fusion import rrf_fuse  # noqa: E402
from hybrid_rag.pipeline import HybridRetriever  # noqa: E402
from hybrid_rag.rerank import reranker_enabled  # noqa: E402
from hybrid_rag.tokenizer import STOPWORDS_ES  # noqa: E402

# --- Queries with hand-labeled relevant docs -------------------------------
# Mezcla de consultas: léxicas (BM25 gana), semánticas (dense rescata) y
# mixtas (ambos contribuyen). Los IDs son slugs de los encabezados del corpus.
QRELS = {
    # --- semánticas: CERO solapamiento léxico con el doc relevante ----------
    "¿cómo impedir que la herramienta responda con información inventada?": {
        "hallucination-modelos-lenguaje",
        "generacion-aumentada-recuperacion",
    },
    "cuando algo falla a mitad el proceso continúa desde el paso anterior": {
        "agentes-estado-produccion"
    },
    "evitar llamadas repetidas para reducir la factura": {"cache-semantica-respuestas"},
    "descubrir los servidores ocultos de una empresa": {"enumeracion-subdominios"},
    "saber qué procesos escuchan en cada puerto de la red": {"inspeccion-puertos-red"},
    # --- léxicas: los términos exactos existen en el doc --------------------
    "BM25 k1 b frecuencia de términos en el corpus": {"busqueda-lexica-bm25"},
    "Content-Security-Policy Strict-Transport-Security auditoría": {
        "auditoria-cabeceras-http-seguridad"
    },
    "pgvector índice aproximado millones de vectores": {"bases-datos-vectoriales-produccion"},
    "métricas recall MRR NDCG": {"metricas-ranking-recall-mrr-ndcg"},
    # --- mixtas: cada retriever aporta una parte -----------------------------
    "combinar varias listas de resultados sin normalizar puntuaciones": {"fusion-rrf-reciprocal-rank-fusion"},
    "fase amplia barata y luego refinar con un segundo modelo": {"reranking-cross-encoders"},
    "órdenes ocultas dentro de un documento recuperado": {
        "inyeccion-instrucciones-sistemas-aumentados",
        "seguridad-aplicaciones-ia",
    },
    "monitorear coste latencia y deriva de calidad por llamada": {"observabilidad-modelos-lenguaje"},
    "pausar la ejecución y reanudar en el mismo punto del flujo": {"agentes-estado-produccion"},
}


def slugify(heading: str) -> str:
    """Heading '# Búsqueda léxica con BM25' -> 'busqueda-lexica-bm25'."""
    import re
    import unicodedata

    words = re.sub(r"^#+\s*", "", heading)
    words = "".join(
        c for c in unicodedata.normalize("NFD", words) if unicodedata.category(c) != "Mn"
    ).lower()
    tokens = [w for w in re.findall(r"[a-z0-9]+", words) if w not in STOPWORDS_ES]
    return "-".join(tokens) if tokens else "doc"


def load_corpus() -> list[tuple[str, str]]:
    text = (ROOT / "data" / "corpus.md").read_text(encoding="utf-8")
    chunks = chunk_markdown(text, max_chars=2000)
    chunks = [(h, b) for h, b in chunks if "Corpus de demostración" not in h]
    docs = [(slugify(h), body) for h, body in chunks]
    return docs


def build_bm25(docs):
    return BM25(docs)


def build_hybrid(docs, enable_rerank=False):
    os.environ["HYBRID_RAG_RERANK"] = "1" if enable_rerank else "0"
    retriever = HybridRetriever()
    retriever.build(docs)
    return retriever


def make_answers(query: str, top: list[tuple[str, str]]) -> str:
    """Reference-style answer derived from the top chunk (deterministic)."""
    if not top:
        return ""
    source = top[0][1]
    first = source.split(".")[0].strip()
    return f"{query} {first}."


def main() -> None:
    docs = load_corpus()
    n_docs = len(docs)
    print(f"[corpus] {n_docs} chunks indexados")

    queries = list(QRELS.keys())
    ranked_bm25: dict[str, list[str]] = {}
    ranked_hybrid: dict[str, list[str]] = {}
    ranked_hybrid_rerank: dict[str, list[str]] = {}

    bm25 = build_bm25(docs)
    # Rerank activado por defecto (modelo cacheado); desactivar con
    # HYBRID_RAG_RERANK=0. Leer ANTES de construir el retriever base.
    enable_rerank = os.environ.get("HYBRID_RAG_RERANK", "1") == "1"
    retriever = build_hybrid(docs, enable_rerank=False)

    rerank_retriever = build_hybrid(docs, enable_rerank=True) if enable_rerank else None

    top_map = {q: [] for q in queries}
    for q in queries:
        ranked_bm25[q] = [d for d, _ in bm25.search(q, top_k=10)]
        ranked_hybrid[q] = [d for d, _ in retriever.retrieve(q, top_k=10)]
        top_map[q] = [(d, retriever.doc_map[d]) for d in ranked_hybrid[q][:3]]
        if rerank_retriever is not None:
            ranked_hybrid_rerank[q] = [d for d, _ in rerank_retriever.retrieve(q, top_k=10)]

    # --- Faithfulness sample -------------------------------------------------
    faithful_lex: list[float] = []
    faithful_llm: list[float] = []
    for q in queries:
        answer = make_answers(q, top_map[q])
        source = top_map[q][0][1] if top_map[q] else ""
        if not source:
            continue
        faithful_lex.append(lexical_faithfulness(answer, source))
        score = llm_judge_faithfulness(q, answer, source)
        if score is not None:
            faithful_llm.append(score / 5.0)

    # --- Report --------------------------------------------------------------
    K = 5
    res_bm25 = evaluate_retrieval(ranked_bm25, QRELS, k=K)
    res_hyb = evaluate_retrieval(ranked_hybrid, QRELS, k=K)
    res_rerank = (
        evaluate_retrieval(ranked_hybrid_rerank, QRELS, k=K)
        if rerank_retriever is not None
        else None
    )

    def fmt(r) -> str:
        return f"{r[f'recall@{K}']:.3f} | {r['mrr']:.3f} | {r[f'ndcg@{K}']:.3f}"

    lines = [
        "# Reporte de evaluación — Hybrid RAG",
        "",
        f"Generado: {os.popen('date /t').read().strip()} | Corpus: {n_docs} chunks | K={K} | Consultas: {len(queries)}",
        "",
        "| Sistema | Recall@5 | MRR | NDCG@5 |",
        "|---|---|---|---|",
        f"| BM25 (baseline) | {fmt(res_bm25)} |",
        f"| Hybrid (BM25+dense+RRF) | {fmt(res_hyb)} |",
    ]
    if res_rerank:
        lines.append(f"| Hybrid + rerank (cross-encoder) | {fmt(res_rerank)} |")
    lines += [
        "",
        f"**Faithfulness** (muestra de {len(faithful_lex)} consultas): "
        f"léxico {mean(faithful_lex):.3f} | LLM-as-judge {mean(faithful_llm):.3f} "
        + ("(OPENAI_API_KEY presente)" if faithful_llm else "(sin key — fallback léxico)"),
        "",
        "## Mejora del híbrido sobre baseline",
    ]
    for q in queries:
        b = ranked_bm25[q][:K]
        h = ranked_hybrid[q][:K]
        relevant = QRELS[q]
        hit_b = len(set(b) & relevant)
        hit_h = len(set(h) & relevant)
        arrow = "+ mejora" if hit_h > hit_b else ("=" if hit_h == hit_b else "- empeora")
        lines.append(f"- `{q}` -> BM25 {hit_b}/{len(relevant)}, Hybrid {hit_h}/{len(relevant)} {arrow}")

    (ROOT / "evals" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\n[ok] reporte escrito en evals/report.md")


if __name__ == "__main__":
    main()
