"""
qrels.py — cross-check exposure against relevance judgments.

Bridges corpus-exposure metrics (coverage / dark matter) and relevance
metrics (recall): no competitor ships this. A dark-matter chunk that is
actually relevant to some query is "lost gold" — the corpus contains
relevant material the retriever never surfaces. This module measures
how much of dark matter is genuinely lost relevant material vs noise.

BEIR / TREC qrels format: {query_id: {doc_id: grade}}. Works with the
standard ProbeResult JSON (save_probe) and a queries.jsonl whose line
order matches hits_per_query (the same contract case_run produces).
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """Load qrels.json: {query_id: {doc_id: grade}}. Grades are coerced to int."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {qid: {did: int(g) for did, g in docs.items()} for qid, docs in raw.items()}


def load_query_ids(path: str) -> list[str]:
    """Load query ids from queries.jsonl (line order = hits_per_query order)."""
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(str(json.loads(line)["id"]))
    return ids


@dataclass
class QrelsValidation:
    """Result of cross-checking a probe run against qrels."""
    n_chunks: int
    n_queries: int
    dark_matter: int
    relevant_in_corpus: int
    dark_and_relevant: int          # "lost gold"
    dark_relevant_pct_of_dark: float
    dark_relevant_pct_of_relevant: float
    qrels_pairs_total: int
    qrels_pairs_in_topk: int
    recall_at_k: float
    dark_relevant_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "n_queries": self.n_queries,
            "dark_matter": self.dark_matter,
            "relevant_in_corpus": self.relevant_in_corpus,
            "dark_and_relevant": self.dark_and_relevant,
            "dark_relevant_pct_of_dark": self.dark_relevant_pct_of_dark,
            "dark_relevant_pct_of_relevant": self.dark_relevant_pct_of_relevant,
            "qrels_pairs_total": self.qrels_pairs_total,
            "qrels_pairs_in_topk": self.qrels_pairs_in_topk,
            "recall_at_k": self.recall_at_k,
            "dark_relevant_ids": self.dark_relevant_ids,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 64,
            "QRELS VALIDATE — dark matter vs relevance",
            "=" * 64,
            f"  Corpus: {self.n_chunks} chunks, queries: {self.n_queries}",
            f"  Dark matter:                {self.dark_matter}",
            f"  Relevant (qrels) in corpus: {self.relevant_in_corpus}",
            "-" * 64,
            f"  Lost gold (dark AND relevant): {self.dark_and_relevant}",
            f"    = {self.dark_relevant_pct_of_dark*100:5.1f}% of dark matter",
            f"    = {self.dark_relevant_pct_of_relevant*100:5.1f}% of relevant chunks",
            "-" * 64,
            f"  Recall@k by qrels: {self.recall_at_k*100:5.1f}% "
            f"({self.qrels_pairs_in_topk}/{self.qrels_pairs_total} pairs)",
            "=" * 64,
        ]
        return "\n".join(lines)


def validate_qrels(probe_path: str, qrels_path: str, queries_path: str) -> QrelsValidation:
    """
    Cross-check a saved probe run against qrels.

    probe_path:   save_probe JSON (freqs + hits_per_query + report).
    qrels_path:   qrels.json {query_id: {doc_id: grade}}.
    queries_path: queries.jsonl whose line order matches hits_per_query.

    Raises ValueError if queries count != hits_per_query count.
    """
    with open(probe_path, encoding="utf-8") as f:
        probe = json.load(f)
    qrels = load_qrels(qrels_path)
    query_ids = load_query_ids(queries_path)

    freqs: dict[str, int] = probe["freqs"]
    hits_per_query: list[list[str]] = probe["hits_per_query"]
    if len(query_ids) != len(hits_per_query):
        raise ValueError(
            f"queries ({len(query_ids)}) != hits_per_query ({len(hits_per_query)}); "
            f"the queries file does not match this probe run"
        )

    corpus_ids = set(freqs)
    dark = {cid for cid, v in freqs.items() if v == 0}

    # chunks relevant to at least one workload query per qrels
    relevant: set[str] = set()
    for qid in query_ids:
        for did in qrels.get(qid, {}):
            if did in corpus_ids:
                relevant.add(did)

    dark_relevant = dark & relevant  # lost gold

    # recall@k by qrels: relevant (query, doc) pairs that landed in top-k
    pairs_total = 0
    pairs_hit = 0
    for qid, hits in zip(query_ids, hits_per_query):
        rel_docs = [d for d in qrels.get(qid, {}) if d in corpus_ids]
        pairs_total += len(rel_docs)
        hit_set = set(hits)
        pairs_hit += sum(1 for d in rel_docs if d in hit_set)

    return QrelsValidation(
        n_chunks=len(corpus_ids),
        n_queries=len(query_ids),
        dark_matter=len(dark),
        relevant_in_corpus=len(relevant),
        dark_and_relevant=len(dark_relevant),
        dark_relevant_pct_of_dark=round(len(dark_relevant) / len(dark), 4) if dark else 0.0,
        dark_relevant_pct_of_relevant=round(len(dark_relevant) / len(relevant), 4) if relevant else 0.0,
        qrels_pairs_total=pairs_total,
        qrels_pairs_in_topk=pairs_hit,
        recall_at_k=round(pairs_hit / pairs_total, 4) if pairs_total else 0.0,
        dark_relevant_ids=sorted(dark_relevant),
    )