"""Run a workload against a vector store and collect raw retrieval exposure."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from retrieval_fairness.metrics import FairnessReport, build_report, retrieval_frequencies
from retrieval_fairness.provenance import ProbeMetadata
from retrieval_fairness.types import Query, VectorStore
from retrieval_fairness.validation import require_positive_int, validate_unique_ids, validate_vector

SCHEMA_VERSION = 3


def _hash_text_parts(parts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        data = part.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def stable_ids_fingerprint(ids: Sequence[str]) -> str:
    """Deterministic SHA-256 over IDs in the supplied physical order."""
    for item in ids:
        if not isinstance(item, str):
            raise ValueError(f"fingerprint IDs must be strings, got {type(item).__name__}")
    return _hash_text_parts(ids)


def stable_set_fingerprint(ids: Sequence[str]) -> str:
    """Deterministic SHA-256 over a logical ID set, independent of row order."""
    values = ids if isinstance(ids, list) else list(ids)
    validate_unique_ids(values, name="fingerprint IDs")
    # Ordered backends (for example pgvector's ORDER BY id) stream directly
    # through the hash without allocating a second sorted million-ID list.
    canonical = (
        values
        if all(values[index - 1] <= values[index] for index in range(1, len(values)))
        else sorted(values)
    )
    return _hash_text_parts(canonical)


def normalize_identity_text(text: str) -> str:
    """Canonical text normalization used only for semantic identity hashes."""
    if not isinstance(text, str):
        raise ValueError("identity content must be text")
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def stable_content_fingerprint(items: Sequence[tuple[str, str]]) -> str:
    """Hash normalized ``(ID, content)`` pairs in canonical ID order."""
    by_id: dict[str, str] = {}
    for item_id, content in items:
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("content fingerprint IDs must be non-empty strings")
        if item_id in by_id:
            raise ValueError(f"content fingerprint contains duplicate ID {item_id!r}")
        by_id[item_id] = normalize_identity_text(content)
    parts: list[str] = []
    for item_id in sorted(by_id):
        parts.extend((item_id, by_id[item_id]))
    return _hash_text_parts(parts)


@dataclass
class ProbeResult:
    """Raw probe observations, semantic identities and a derived report."""

    freqs: dict[str, int]
    hits_per_query: list[list[str]] = field(default_factory=list)
    query_ids: list[str] = field(default_factory=list)
    report: FairnessReport | None = None
    schema_version: int = SCHEMA_VERSION

    # v1/v2 compatibility aliases: ordered IDs only.
    corpus_fingerprint: str | None = None
    workload_fingerprint: str | None = None

    # Schema-v3 identity model.
    workload_ids_fingerprint: str | None = None
    workload_content_fingerprint: str | None = None
    workload_revision: str | None = None
    corpus_set_fingerprint: str | None = None
    corpus_order_fingerprint: str | None = None
    corpus_content_fingerprint: str | None = None
    corpus_revision: str | None = None
    index_mapping_fingerprint: str | None = None

    metadata: dict[str, object] = field(default_factory=dict)


def _optional_content_fingerprint(items: list[tuple[str, str]]) -> str | None:
    if not items or any(not normalize_identity_text(content) for _, content in items):
        return None
    return stable_content_fingerprint(items)


def probe(
    store: VectorStore,
    queries: list[Query],
    top_k: int = 10,
    corpus_ids: list[str] | None = None,
    *,
    corpus_texts: dict[str, str] | None = None,
    workload_revision: str | None = None,
    corpus_revision: str | None = None,
    embedder: str | None = None,
    embedder_revision: str | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    report_detail: str = "full",
) -> ProbeResult:
    """Run queries and derive a report with reproducible input identities."""
    require_positive_int(top_k, "top_k")
    query_ids = [str(query.id) for query in queries]
    validate_unique_ids(query_ids, name="query_ids")
    for index, query in enumerate(queries):
        validate_vector(query.vector, name=f"queries[{index}].vector")

    corpus_content: list[tuple[str, str]] = []
    if corpus_ids is None:
        chunks = list(store.list_chunks())
        corpus_ids = [str(chunk.id) for chunk in chunks]
        corpus_content = [(str(chunk.id), chunk.text) for chunk in chunks]
    else:
        corpus_ids = [str(chunk_id) for chunk_id in corpus_ids]
    validate_unique_ids(corpus_ids, name="corpus_ids")
    corpus_set = set(corpus_ids)
    if corpus_texts is not None:
        normalized_texts = {str(chunk_id): text for chunk_id, text in corpus_texts.items()}
        if set(normalized_texts) != corpus_set:
            raise ValueError("corpus_texts IDs must exactly match corpus_ids")
        corpus_content = [(chunk_id, normalized_texts[chunk_id]) for chunk_id in corpus_ids]

    hits_per_query: list[list[str]] = []
    for query in queries:
        hits = [str(hit.chunk_id) for hit in store.search(query.vector, top_k)]
        validate_unique_ids(hits, name=f"hits for query {query.id!r}")
        if len(hits) > top_k:
            raise ValueError(
                f"store returned {len(hits)} hits for query {query.id!r}, exceeding top_k={top_k}"
            )
        unknown = [chunk_id for chunk_id in hits if chunk_id not in corpus_set]
        if unknown:
            raise ValueError(f"store returned hit IDs outside corpus for query {query.id!r}: {unknown[:5]!r}")
        hits_per_query.append(hits)

    freqs = retrieval_frequencies(hits_per_query, corpus_ids)
    report = build_report(freqs, n_queries=len(queries), top_k=top_k, detail=report_detail)
    query_content = [(str(query.id), query.text) for query in queries]
    metadata = ProbeMetadata.for_run(
        store,
        top_k=top_k,
        embedder=embedder,
        embedder_revision=embedder_revision,
        corpus_revision=corpus_revision,
        workload_revision=workload_revision,
        run_id=run_id,
        git_commit=git_commit,
    ).to_dict()
    index_mapping = getattr(store, "index_mapping_fingerprint", None)
    if callable(index_mapping):
        index_mapping = index_mapping()
    if index_mapping is not None and not isinstance(index_mapping, str):
        raise ValueError("store index_mapping_fingerprint must be a string")

    return ProbeResult(
        freqs=freqs,
        hits_per_query=hits_per_query,
        query_ids=query_ids,
        report=report,
        corpus_fingerprint=stable_ids_fingerprint(corpus_ids),
        workload_fingerprint=stable_ids_fingerprint(query_ids),
        workload_ids_fingerprint=stable_set_fingerprint(query_ids),
        workload_content_fingerprint=_optional_content_fingerprint(query_content),
        workload_revision=workload_revision,
        corpus_set_fingerprint=stable_set_fingerprint(corpus_ids),
        corpus_order_fingerprint=stable_ids_fingerprint(corpus_ids),
        corpus_content_fingerprint=_optional_content_fingerprint(corpus_content),
        corpus_revision=corpus_revision,
        index_mapping_fingerprint=index_mapping,
        metadata=metadata,
    )
