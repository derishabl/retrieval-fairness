"""Run a workload against a vector store and collect retrieval frequency."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from collections.abc import Sequence

from retrieval_fairness.metrics import FairnessReport, build_report, retrieval_frequencies
from retrieval_fairness.types import Query, VectorStore
from retrieval_fairness.validation import (
    require_positive_int,
    validate_unique_ids,
    validate_vector,
)

SCHEMA_VERSION = 2


def stable_ids_fingerprint(ids: Sequence[str]) -> str:
    """Return a deterministic, process-independent SHA-256 over ordered IDs."""
    h = hashlib.sha256()
    for item in ids:
        if not isinstance(item, str):
            raise ValueError(f"fingerprint IDs must be strings, got {type(item).__name__}")
        data = item.encode("utf-8")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return f"sha256:{h.hexdigest()}"


@dataclass
class ProbeResult:
    """Raw probe observations and a report derived from those observations."""

    freqs: dict[str, int]
    hits_per_query: list[list[str]] = field(default_factory=list)
    query_ids: list[str] = field(default_factory=list)
    report: FairnessReport | None = None
    schema_version: int = SCHEMA_VERSION
    corpus_fingerprint: str | None = None
    workload_fingerprint: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def probe(
    store: VectorStore,
    queries: list[Query],
    top_k: int = 10,
    corpus_ids: list[str] | None = None,
) -> ProbeResult:
    """Run queries, retain raw hits, and derive a fairness report."""
    require_positive_int(top_k, "top_k")
    query_ids = [str(q.id) for q in queries]
    validate_unique_ids(query_ids, name="query_ids")
    for index, query in enumerate(queries):
        validate_vector(query.vector, name=f"queries[{index}].vector")

    if corpus_ids is None:
        corpus_ids = [str(c.id) for c in store.list_chunks()]
    else:
        corpus_ids = [str(cid) for cid in corpus_ids]
    validate_unique_ids(corpus_ids, name="corpus_ids")
    corpus_set = set(corpus_ids)

    hits_per_query: list[list[str]] = []
    for query in queries:
        hits = [str(hit.chunk_id) for hit in store.search(query.vector, top_k)]
        validate_unique_ids(hits, name=f"hits for query {query.id!r}")
        if len(hits) > top_k:
            raise ValueError(
                f"store returned {len(hits)} hits for query {query.id!r}, exceeding top_k={top_k}"
            )
        unknown = [cid for cid in hits if cid not in corpus_set]
        if unknown:
            raise ValueError(f"store returned hit IDs outside corpus for query {query.id!r}: {unknown[:5]!r}")
        hits_per_query.append(hits)

    freqs = retrieval_frequencies(hits_per_query, corpus_ids)
    report = build_report(freqs, n_queries=len(queries), top_k=top_k)
    return ProbeResult(
        freqs=freqs,
        hits_per_query=hits_per_query,
        query_ids=query_ids,
        report=report,
        corpus_fingerprint=stable_ids_fingerprint(corpus_ids),
        workload_fingerprint=stable_ids_fingerprint(query_ids),
    )
