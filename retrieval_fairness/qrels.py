"""Cross-check probe exposure against positive qrels judgments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from retrieval_fairness.serialize import load_probe
from retrieval_fairness.validation import require_integral, require_positive_int, validate_unique_ids


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("qrels must be an object")
    output: dict[str, dict[str, int]] = {}
    for query_id, docs in raw.items():
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("qrels query IDs must be non-empty strings")
        if not isinstance(docs, dict):
            raise ValueError(f"qrels[{query_id!r}] must be an object")
        judgments: dict[str, int] = {}
        for doc_id, grade in docs.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"qrels[{query_id!r}] document IDs must be non-empty strings")
            judgments[doc_id] = require_integral(grade, f"qrels[{query_id!r}][{doc_id!r}]")
        output[query_id] = judgments
    return output


def load_query_ids(path: str) -> list[str]:
    ids: list[str] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                ids.append(str(json.loads(line)["id"]))
    validate_unique_ids(ids, name="query IDs")
    return ids


@dataclass
class QrelsValidation:
    n_chunks: int
    n_queries: int
    dark_matter: int
    relevant_in_corpus: int
    dark_and_relevant: int
    dark_relevant_pct_of_dark: float
    dark_relevant_pct_of_relevant: float
    qrels_pairs_total: int
    qrels_pairs_in_topk: int
    micro_recall_at_k: float
    dark_relevant_ids: list[str] = field(default_factory=list)
    macro_recall_at_k: float = 0.0
    per_query_recall: dict[str, float] = field(default_factory=dict)
    queries_with_relevant_docs: int = 0

    @property
    def recall_at_k(self) -> float:
        """Read-only compatibility alias for micro recall."""
        return self.micro_recall_at_k

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
            "micro_recall_at_k": self.micro_recall_at_k,
            "macro_recall_at_k": self.macro_recall_at_k,
            "per_query_recall": self.per_query_recall,
            "queries_with_relevant_docs": self.queries_with_relevant_docs,
            "dark_relevant_ids": self.dark_relevant_ids,
        }

    def __str__(self) -> str:
        return "\n".join(
            [
                "=" * 64,
                "QRELS VALIDATE — dark matter vs relevance",
                "=" * 64,
                f"  Corpus: {self.n_chunks} chunks, queries: {self.n_queries}",
                f"  Dark matter:                {self.dark_matter}",
                f"  Relevant (qrels) in corpus: {self.relevant_in_corpus}",
                "-" * 64,
                f"  Lost gold (dark AND relevant): {self.dark_and_relevant}",
                f"    = {self.dark_relevant_pct_of_dark * 100:5.1f}% of dark matter",
                f"    = {self.dark_relevant_pct_of_relevant * 100:5.1f}% of relevant chunks",
                "-" * 64,
                f"  Micro recall@k: {self.micro_recall_at_k * 100:5.1f}% "
                f"({self.qrels_pairs_in_topk}/{self.qrels_pairs_total} pairs)",
                f"  Macro recall@k: {self.macro_recall_at_k * 100:5.1f}% "
                f"({self.queries_with_relevant_docs} queries with relevant docs)",
                "=" * 64,
            ]
        )


def validate_qrels(
    probe_path: str,
    qrels_path: str,
    queries_path: str | None = None,
    *,
    min_relevance_grade: int = 1,
) -> QrelsValidation:
    """Validate positive qrels against a probe.

    A judgment is relevant iff ``grade >= min_relevance_grade``. Schema-v2
    probes carry their own query IDs; ``queries_path`` is only required for
    legacy probes and acts as an additional exact-order check for v2.
    """
    require_positive_int(min_relevance_grade, "min_relevance_grade")
    result = load_probe(probe_path, strict_integrity=False)
    qrels = load_qrels(qrels_path)

    if result.query_ids:
        query_ids = result.query_ids
        if queries_path is not None:
            external_ids = load_query_ids(queries_path)
            if external_ids != query_ids:
                raise ValueError("external queries IDs/order does not match probe query_ids")
    else:
        if queries_path is None:
            raise ValueError("--queries is required for a legacy probe without query IDs")
        query_ids = load_query_ids(queries_path)
        if len(query_ids) != len(result.hits_per_query):
            raise ValueError(
                f"queries ({len(query_ids)}) != hits_per_query "
                f"({len(result.hits_per_query)}); the queries file does not match this probe run"
            )

    validate_unique_ids(query_ids, name="probe query IDs")
    corpus_ids = set(result.freqs)
    dark = {chunk_id for chunk_id, count in result.freqs.items() if count == 0}

    relevant: set[str] = set()
    relevant_by_query: dict[str, set[str]] = {}
    for query_id in query_ids:
        docs = {
            doc_id
            for doc_id, grade in qrels.get(query_id, {}).items()
            if grade >= min_relevance_grade and doc_id in corpus_ids
        }
        relevant_by_query[query_id] = docs
        relevant.update(docs)

    dark_relevant = dark & relevant
    pairs_total = 0
    pairs_hit = 0
    per_query: dict[str, float] = {}
    for query_id, hits in zip(query_ids, result.hits_per_query):
        docs = relevant_by_query[query_id]
        if not docs:
            continue
        found = len(docs & set(hits))
        pairs_total += len(docs)
        pairs_hit += found
        per_query[query_id] = found / len(docs)

    micro = pairs_hit / pairs_total if pairs_total else 0.0
    macro = sum(per_query.values()) / len(per_query) if per_query else 0.0
    micro = round(micro, 4)
    macro = round(macro, 4)
    return QrelsValidation(
        n_chunks=len(corpus_ids),
        n_queries=len(query_ids),
        dark_matter=len(dark),
        relevant_in_corpus=len(relevant),
        dark_and_relevant=len(dark_relevant),
        dark_relevant_pct_of_dark=(round(len(dark_relevant) / len(dark), 4) if dark else 0.0),
        dark_relevant_pct_of_relevant=(round(len(dark_relevant) / len(relevant), 4) if relevant else 0.0),
        qrels_pairs_total=pairs_total,
        qrels_pairs_in_topk=pairs_hit,
        micro_recall_at_k=micro,
        dark_relevant_ids=sorted(dark_relevant),
        macro_recall_at_k=macro,
        per_query_recall={key: round(value, 4) for key, value in per_query.items()},
        queries_with_relevant_docs=len(per_query),
    )
