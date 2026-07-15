"""
probe_precomputed.py — лёгкая локальная часть полного прогона: FAISS + probe
по заранее посчитанным векторам (из scripts/colab_nq_bge_encode.py).

На 16 GB RAM полный NQ (260k × 384 float32 ≈ 400 МБ) обрабатывается
за минуты — тяжёлым был только encode, он сделан на GPU.

Использование:
  python scripts/probe_precomputed.py --dir nq_full --top-k 10 \
      --out cases/nq_full_bge --html cases/nq_full_bge.html
  # затем qrels-валидация:
  python scripts/qrels_validate.py --probe cases/nq_full_bge.json \
      --qrels nq_full/qrels.json --queries nq_full/queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_fairness.adapters.faiss import FaissAdapter, build_flat_index
from retrieval_fairness.probe import probe
from retrieval_fairness.serialize import save_probe
from retrieval_fairness.types import Query


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe on precomputed vectors")
    ap.add_argument(
        "--dir",
        required=True,
        help="каталог с corpus_vecs.npy, corpus_ids.json, query_vecs.npy, queries.jsonl",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", required=True, help="prefix для .json/.faiss")
    ap.add_argument("--html", help="HTML dashboard (на больших корпусах PCA медленный)")
    args = ap.parse_args()

    d = args.dir
    chunk_arr = np.load(os.path.join(d, "corpus_vecs.npy")).astype("float32")
    with open(os.path.join(d, "corpus_ids.json"), encoding="utf-8") as f:
        ids = json.load(f)
    q_arr = np.load(os.path.join(d, "query_vecs.npy")).astype("float32")
    q_meta = []
    with open(os.path.join(d, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q_meta.append(json.loads(line))
    assert len(ids) == chunk_arr.shape[0], "corpus_ids не соответствует corpus_vecs"
    assert len(q_meta) == q_arr.shape[0], "queries.jsonl не соответствует query_vecs"
    print(f"corpus: {chunk_arr.shape}, queries: {q_arr.shape}, top-k: {args.top_k}")

    # нормализация под cosine-via-IP (encode мог уже нормализовать — идемпотентно)
    for arr in (chunk_arr, q_arr):
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.where(norms < 1e-12, 1e-12, norms)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    idx_path, ids_map = args.out + ".faiss", args.out + ".ids.json"
    t0 = time.time()
    build_flat_index(chunk_arr.tolist(), ids, idx_path, ids_map, metric="ip")
    print(f"index built in {time.time() - t0:.0f}s")

    queries = [
        Query(id=str(m["id"]), vector=v.tolist(), text=m.get("text", "")) for m, v in zip(q_meta, q_arr)
    ]
    store = FaissAdapter(idx_path, ids_map)
    t0 = time.time()
    result = probe(store, queries, top_k=args.top_k, corpus_ids=ids)
    print(f"probe done in {time.time() - t0:.0f}s")
    print(result.report)

    save_probe(result, args.out + ".json")
    print(f"baseline saved: {args.out}.json")

    if args.html:
        from retrieval_fairness.dashboard import render_dashboard

        render_dashboard(result, args.html, chunks_vectors=chunk_arr.tolist(), chunk_ids=ids)
        print(f"dashboard saved: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
