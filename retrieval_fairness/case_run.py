"""
case_run.py — прогон retrieval-fairness на корпусе для кейса (Шаг 9).

Связывает: Embedder (TF-IDF/MiniLM) -> векторы -> FAISS-индекс ->
probe -> baseline JSON + dashboard. Один entrypoint для smoke и полного
прогона на NQ/MS MARCO.

Использование:
  python -m retrieval_fairness.case_run \
      --corpus data/nq/corpus.jsonl --queries data/nq/queries.jsonl \
      --embedder tfidf --top-k 10 \
      --out cases/nq_tfidf --html cases/nq_tfidf.html
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from retrieval_fairness.adapters.faiss import FaissAdapter, build_flat_index
from retrieval_fairness.embedders import get_embedder
from retrieval_fairness.probe import probe
from retrieval_fairness.serialize import save_probe
from retrieval_fairness.types import Chunk, Query


def load_corpus(path: str) -> list[Chunk]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out.append(Chunk(id=r["id"], text=r.get("text", r["id"]), vector=None))
    return out


def load_queries(path: str) -> list[dict]:
    """Загрузить запросы как plain dicts (вектор посчитает эмбеддер)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_case(
    corpus: list[Chunk],
    queries: list[dict],
    embedder_name: str,
    top_k: int,
    out_prefix: str,
    html_path: str | None = None,
    max_features: int | None = None,
) -> None:
    """Полный цикл: embed -> FAISS -> probe -> save + dashboard."""
    print(f"[case] embedder={embedder_name} corpus={len(corpus)} queries={len(queries)} top-k={top_k}")

    # 1. embed
    kw = {}
    if embedder_name == "tfidf" and max_features:
        kw["max_features"] = max_features
    emb = get_embedder(embedder_name, **kw)
    texts = [c.text for c in corpus]
    q_texts = [q.get("text", q["id"]) for q in queries]
    # fit ТОЛЬКО на корпусе — запросы не входят в vocabulary.
    # Иначе векторы корпуса зависят от набора запросов, и diff между прогонами
    # с разными запросами портится (silent wrong result). Для dense (MiniLM/BGE)
    # fit — no-op, так что поведение не меняется.
    emb.fit(texts)
    print("[case] encoding corpus...")
    chunk_vecs = emb.encode(texts)
    print("[case] encoding queries...")
    query_vecs = emb.encode(q_texts)
    queries_vec = [
        Query(id=q["id"], vector=v.tolist(), text=q.get("text", "")) for q, v in zip(queries, query_vecs)
    ]

    # 2. FAISS index
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    idx_path = out_prefix + ".faiss"
    ids_map = out_prefix + ".ids.json"
    ids = [c.id for c in corpus]
    # normalize for cosine via IP
    chunk_arr = np.array(chunk_vecs, dtype="float32")
    norms = np.linalg.norm(chunk_arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    chunk_arr = chunk_arr / norms
    build_flat_index(
        chunk_arr.tolist(),
        ids,
        idx_path,
        ids_map,
        metric="ip",
        normalized=True,
    )

    # 3. probe
    store = FaissAdapter(idx_path, ids_map)
    result = probe(
        store,
        queries_vec,
        top_k=top_k,
        corpus_texts={chunk.id: chunk.text for chunk in corpus},
        embedder=embedder_name,
    )
    print(result.report)

    # 4. save
    save_probe(result, out_prefix + ".json")
    print(f"[case] baseline saved: {out_prefix}.json")

    # 5. dashboard (with vectors for PCA)
    if html_path:
        from retrieval_fairness.dashboard import render_dashboard

        render_dashboard(result, html_path, chunks_vectors=chunk_arr.tolist(), chunk_ids=ids)
        print(f"[case] dashboard saved: {html_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--embedder", default="tfidf", choices=["tfidf", "minilm", "fastembed", "bge"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", required=True, help="prefix for outputs (no extension)")
    ap.add_argument("--html", help="path for HTML dashboard")
    ap.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="tfidf: ограничить словарь (обязательно для больших корпусов, иначе OOM)",
    )
    args = ap.parse_args()
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    if args.embedder == "tfidf" and args.max_features is None and len(corpus) > 20000:
        print(
            f"WARNING: tfidf на {len(corpus)} чанках без --max-features — риск OOM "
            "(dense-матрица под FAISS). Рекомендуется --max-features 4096."
        )
    run_case(corpus, queries, args.embedder, args.top_k, args.out, args.html, max_features=args.max_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
