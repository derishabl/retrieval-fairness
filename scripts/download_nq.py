"""
scripts/download_nq.py — загрузка BEIR NQ (HuggingFace BeIR/nq) для кейса (Шаг 9).

BEIR-формат: готовые queries/corpus/qrels. Скачивает в data/nq/.
data/ в .gitignore (большой).

Использование:
  python scripts/download_nq.py --out data/nq --sample-corpus 5000 --sample-queries 500
  (без --sample-* — полный корпус NQ)

Зависимости: datasets (HuggingFace).
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser(description="Download BEIR NQ (HuggingFace BeIR/nq)")
    ap.add_argument("--out", default="data/nq")
    ap.add_argument("--sample-corpus", type=int, default=0, help="0 = full corpus; else sample N chunks")
    ap.add_argument("--sample-queries", type=int, default=0, help="0 = all queries; else sample N")
    ap.add_argument("--split", default="test", help="queries split: test/dev")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading BEIR NQ -> {args.out} (split={args.split})")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", flush=True)
        return 2

    # corpus
    print("loading corpus...")
    corpus_ds = load_dataset("BeIR/nq", "corpus", split="corpus")
    if args.sample_corpus > 0:
        corpus_ds = corpus_ds.select(range(min(args.sample_corpus, len(corpus_ds))))
    with open(os.path.join(args.out, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for r in corpus_ds:
            title = r.get("title", "")
            text = r.get("text", "")
            full = f"{title}\n{text}" if title else text
            f.write(json.dumps({"id": str(r["_id"]), "text": full}, ensure_ascii=False) + "\n")
    print(f"  corpus: {len(corpus_ds)} chunks")

    # queries (тексты — в BeIR/nq config 'queries')
    print("loading queries...")
    qrels_ds = load_dataset("BeIR/nq-qrels", split=args.split)
    queries_ds = load_dataset("BeIR/nq", "queries", split="queries")
    # индекс по query-id -> text
    qid_to_text = {str(r["_id"]): r.get("text", "") for r in queries_ds}
    # qrels: query-id -> {corpus-id: score}
    qrels: dict[str, dict[str, int]] = {}
    qids_seen: list[str] = []
    seen = set()
    for r in qrels_ds:
        qid = str(r["query-id"])
        qrels.setdefault(qid, {})[str(r["corpus-id"])] = int(r["score"])
        if qid not in seen:
            seen.add(qid)
            qids_seen.append(qid)
    if args.sample_queries > 0:
        qids_seen = qids_seen[: args.sample_queries]
        qrels = {k: v for k, v in qrels.items() if k in set(qids_seen)}
    with open(os.path.join(args.out, "queries.jsonl"), "w", encoding="utf-8") as f:
        for qid in qids_seen:
            txt = qid_to_text.get(qid, "")
            f.write(json.dumps({"id": qid, "text": txt}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "qrels.json"), "w", encoding="utf-8") as f:
        json.dump(qrels, f)
    print(f"  queries: {len(qids_seen)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
