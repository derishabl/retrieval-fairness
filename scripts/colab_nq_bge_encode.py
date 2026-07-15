"""
colab_nq_bge_encode.py — тяжёлая часть полного NQ-прогона: encode на GPU (Colab).

Зачем: BGE-small на CPU — ~15 ч на полный NQ (260k чанков); на бесплатном
Colab T4 GPU — ~30–60 мин. Скрипт выполняет ТОЛЬКО тяжёлое (скачать корпус,
закодировать), результат — компактные артефакты (~400 МБ float32), которые
скачиваются и дообрабатываются локально за минуты:

  python scripts/probe_precomputed.py --vectors nq_full/corpus_vecs.npy ...

Использование в Colab (Runtime -> Change runtime type -> T4 GPU):
  1. Загрузить этот файл + scripts/download_nq.py в Colab (или весь repo).
  2. !pip install -q datasets sentence-transformers
  3. !python colab_nq_bge_encode.py --out nq_full
     (--sample-corpus 50000 для промежуточного прогона)
  4. Скачать nq_full/*.npy, nq_full/*.jsonl (files.download или Drive).

Использует sentence-transformers (не fastembed): на GPU torch-бэкенд
быстрее и стабильнее; модель та же BAAI/bge-small-en-v1.5, косинус на
нормализованных векторах — совместимо с локальным fastembed-прогоном
по семантике, но НЕ бит-в-бит (разные рантаймы) — для diff с локальным
baseline это ок (diff сравнивает exposure, не векторы).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Encode BEIR NQ with BGE-small on GPU (Colab)")
    ap.add_argument("--out", default="nq_full")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--sample-corpus", type=int, default=0, help="0 = полный корпус")
    ap.add_argument("--sample-queries", type=int, default=0, help="0 = все запросы test split")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    dev = str(getattr(model, "device", "?"))
    print(f"model={args.model} device={dev}")
    if "cuda" not in dev:
        print(
            "WARNING: GPU не виден — encode будет медленным. Colab: Runtime -> Change runtime type -> T4 GPU."
        )

    # --- корпус ---
    print("loading corpus...")
    corpus_ds = load_dataset("BeIR/nq", "corpus", split="corpus")
    if args.sample_corpus > 0:
        corpus_ds = corpus_ds.select(range(min(args.sample_corpus, len(corpus_ds))))
    ids, texts = [], []
    for r in corpus_ds:
        ids.append(str(r["_id"]))
        title = r.get("title", "")
        texts.append(f"{title}\n{r.get('text', '')}" if title else r.get("text", ""))
    print(f"corpus: {len(ids)} chunks")

    t0 = time.time()
    vecs = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    print(f"corpus encoded in {time.time() - t0:.0f}s")
    np.save(os.path.join(args.out, "corpus_vecs.npy"), np.asarray(vecs, dtype=np.float32))
    with open(os.path.join(args.out, "corpus_ids.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)

    # --- запросы + qrels ---
    print("loading queries/qrels...")
    qrels_ds = load_dataset("BeIR/nq-qrels", split="test")
    queries_ds = load_dataset("BeIR/nq", "queries", split="queries")
    qid_to_text = {str(r["_id"]): r.get("text", "") for r in queries_ds}
    qrels: dict[str, dict[str, int]] = {}
    qids: list[str] = []
    for r in qrels_ds:
        qid = str(r["query-id"])
        if qid not in qrels:
            qrels[qid] = {}
            qids.append(qid)
        qrels[qid][str(r["corpus-id"])] = int(r["score"])
    if args.sample_queries > 0:
        qids = qids[: args.sample_queries]
        qrels = {q: qrels[q] for q in qids}
    q_texts = [qid_to_text.get(q, q) for q in qids]
    print(f"queries: {len(qids)}")

    q_vecs = model.encode(
        q_texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True
    )
    np.save(os.path.join(args.out, "query_vecs.npy"), np.asarray(q_vecs, dtype=np.float32))
    with open(os.path.join(args.out, "queries.jsonl"), "w", encoding="utf-8") as f:
        for qid, txt in zip(qids, q_texts):
            f.write(json.dumps({"id": qid, "text": txt}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "qrels.json"), "w", encoding="utf-8") as f:
        json.dump(qrels, f, ensure_ascii=False)

    total_mb = sum(os.path.getsize(os.path.join(args.out, n)) for n in os.listdir(args.out)) / 2**20
    print(f"done -> {args.out}/ ({total_mb:.0f} MB). Скачайте каталог и запустите локально:")
    print(
        f"  python scripts/probe_precomputed.py --dir {args.out} --top-k 10 "
        f"--out cases/nq_full_bge --html cases/nq_full_bge.html"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
