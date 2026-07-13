"""Versioned, integrity-checked serialization for :class:`ProbeResult`."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from retrieval_fairness import __version__
from retrieval_fairness.metrics import FairnessReport, build_report, retrieval_frequencies
from retrieval_fairness.probe import ProbeResult, SCHEMA_VERSION, stable_ids_fingerprint
from retrieval_fairness.validation import require_non_negative_int, validate_unique_ids

_REPORT_METRICS = (
    "coverage_pct",
    "dark_matter_pct",
    "gini",
    "hub_capture_top5",
    "hub_capture_top10",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _open_text(path: str, mode: str) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _validate_raw(freqs: dict[str, int], hits: list[list[str]], query_ids: list[str], top_k: int) -> None:
    validate_unique_ids(list(freqs), name="corpus IDs")
    validate_unique_ids(query_ids, name="query_ids")
    if len(query_ids) != len(hits):
        raise ValueError(f"len(query_ids)={len(query_ids)} != len(hits_per_query)={len(hits)}")
    for chunk_id, value in freqs.items():
        require_non_negative_int(value, f"frequency[{chunk_id!r}]")
    for index, row in enumerate(hits):
        if not isinstance(row, list) or any(not isinstance(cid, str) for cid in row):
            raise ValueError(f"hits_per_query[{index}] must be a list of string IDs")
        validate_unique_ids(row, name=f"hits_per_query[{index}]")
        if len(row) > top_k:
            raise ValueError(f"hits_per_query[{index}] has {len(row)} hits, exceeding top_k={top_k}")
    rebuilt = retrieval_frequencies(hits, list(freqs))
    if rebuilt != freqs:
        changed = [cid for cid in freqs if rebuilt[cid] != freqs[cid]][:5]
        raise ValueError(f"frequencies do not match hits_per_query; examples: {changed!r}")


def probe_to_json(result: ProbeResult) -> dict[str, Any]:
    """Convert a valid result to baseline schema v2."""
    if result.report is None:
        raise ValueError("ProbeResult.report is required for serialization")
    query_ids = list(result.query_ids)
    if not query_ids and result.hits_per_query:
        raise ValueError("schema v2 serialization requires query_ids")
    _validate_raw(result.freqs, result.hits_per_query, query_ids, result.report.top_k)

    corpus_fingerprint = result.corpus_fingerprint or stable_ids_fingerprint(list(result.freqs))
    workload_fingerprint = result.workload_fingerprint or stable_ids_fingerprint(query_ids)
    expected_corpus = stable_ids_fingerprint(list(result.freqs))
    expected_workload = stable_ids_fingerprint(query_ids)
    if corpus_fingerprint != expected_corpus or workload_fingerprint != expected_workload:
        raise ValueError("ProbeResult fingerprints do not match its raw IDs")

    metadata = dict(result.metadata)
    metadata.setdefault("created_at", _utc_now())
    metadata.setdefault("store", None)
    metadata.setdefault("embedder", None)
    metadata["top_k"] = result.report.top_k
    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "query_ids": query_ids,
        "corpus_fingerprint": corpus_fingerprint,
        "workload_fingerprint": workload_fingerprint,
        "metadata": metadata,
        "freqs": result.freqs,
        "hits_per_query": result.hits_per_query,
        "report": result.report.to_dict(),
    }


def save_probe(result: ProbeResult, path: str, *, compress: bool = False) -> None:
    """Save a full baseline. ``compress=True`` writes gzip data."""
    target = str(path)
    if compress and not target.endswith(".gz"):
        target += ".gz"
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with _open_text(target, "w") as file:
        json.dump(probe_to_json(result), file, ensure_ascii=False, indent=2)


def _parse_freqs(value: object, *, strict: bool) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("freqs must be an object")
    out: dict[str, int] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("frequency IDs must be non-empty strings")
        if strict and (isinstance(raw, bool) or not isinstance(raw, int)):
            raise ValueError(f"frequency[{key!r}] must be an integer")
        try:
            parsed = int(raw)  # legacy compatibility in non-strict mode
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frequency[{key!r}] must be an integer") from exc
        require_non_negative_int(parsed, f"frequency[{key!r}]")
        out[key] = parsed
    return out


def _report_matches(saved: dict[str, Any], rebuilt: FairnessReport) -> bool:
    exact = {"n_chunks": rebuilt.n_chunks, "n_queries": rebuilt.n_queries, "top_k": rebuilt.top_k}
    for key, value in exact.items():
        if key in saved and saved[key] != value:
            return False
    for key in _REPORT_METRICS:
        if key in saved:
            try:
                if round(float(saved[key]), 4) != round(float(getattr(rebuilt, key)), 4):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def load_probe(path: str, *, strict_integrity: bool = True) -> ProbeResult:
    """Load a baseline and always rebuild metrics from raw frequencies.

    Schema-v2 raw invariants are always enforced. For legacy files,
    ``strict_integrity=False`` permits inconsistent/missing hit data while still
    validating frequencies and rebuilding the report from them.
    """
    with _open_text(path, "r") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("baseline root must be a JSON object")

    schema_version = data.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("schema_version must be an integer")
    if schema_version not in (1, SCHEMA_VERSION):
        raise ValueError(f"unsupported baseline schema_version={schema_version}")
    legacy = schema_version == 1

    freqs = _parse_freqs(data.get("freqs"), strict=strict_integrity)
    raw_hits = data.get("hits_per_query", [])
    if not isinstance(raw_hits, list):
        raise ValueError("hits_per_query must be a list")
    hits: list[list[str]] = []
    for index, row in enumerate(raw_hits):
        if not isinstance(row, list) or any(not isinstance(cid, str) for cid in row):
            raise ValueError(f"hits_per_query[{index}] must be a list of string IDs")
        hits.append(list(row))

    raw_query_ids = data.get("query_ids", [])
    if not isinstance(raw_query_ids, list) or any(not isinstance(qid, str) for qid in raw_query_ids):
        raise ValueError("query_ids must be a list of strings")
    query_ids = list(raw_query_ids)
    if not legacy and "query_ids" not in data:
        raise ValueError("schema v2 requires query_ids")

    report_data = data.get("report", {})
    if not isinstance(report_data, dict):
        raise ValueError("report must be an object")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    metadata = dict(metadata)
    if legacy:
        metadata["legacy_schema"] = True
        if not query_ids:
            metadata["legacy_positional_alignment"] = True

    top_k_raw = metadata.get("top_k", report_data.get("top_k"))
    if top_k_raw is None:
        top_k_raw = max((len(row) for row in hits), default=1)
    if isinstance(top_k_raw, bool):
        raise ValueError("top_k must be a positive integer")
    try:
        top_k = int(top_k_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be a positive integer") from exc
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    enforce_raw = not legacy or strict_integrity
    if enforce_raw:
        effective_ids = query_ids if query_ids else [f"legacy:{i}" for i in range(len(hits))]
        _validate_raw(freqs, hits, effective_ids, top_k)
    else:
        for index, row in enumerate(hits):
            if len(row) > top_k:
                metadata["legacy_integrity_relaxed"] = True
                break

    n_queries = len(query_ids) if query_ids else len(hits)
    rebuilt_report = build_report(freqs, n_queries=n_queries, top_k=top_k)
    if strict_integrity and report_data and not _report_matches(report_data, rebuilt_report):
        raise ValueError("saved report does not match metrics rebuilt from raw data")

    corpus_fingerprint = data.get("corpus_fingerprint")
    workload_fingerprint = data.get("workload_fingerprint")
    if not legacy:
        expected_corpus = stable_ids_fingerprint(list(freqs))
        expected_workload = stable_ids_fingerprint(query_ids)
        if corpus_fingerprint != expected_corpus:
            raise ValueError("corpus_fingerprint does not match corpus IDs")
        if workload_fingerprint != expected_workload:
            raise ValueError("workload_fingerprint does not match query IDs")

    return ProbeResult(
        freqs=freqs,
        hits_per_query=hits,
        query_ids=query_ids,
        report=rebuilt_report,
        schema_version=schema_version,
        corpus_fingerprint=corpus_fingerprint,
        workload_fingerprint=workload_fingerprint,
        metadata=metadata,
    )


def probe_summary_to_json(result: ProbeResult, *, max_exported_dark_ids: int = 1000) -> dict[str, Any]:
    """Return an exact-count compact summary that cannot be used as raw input."""
    require_non_negative_int(max_exported_dark_ids, "max_exported_dark_ids")
    full = probe_to_json(result)
    report = dict(full["report"])
    dark_ids = list(report.get("dark_matter_ids", []))
    exported = dark_ids[:max_exported_dark_ids]
    report["dark_matter_ids"] = exported
    report["dark_matter_count"] = len(dark_ids)
    report["dark_matter_ids_exported"] = len(exported)
    report["dark_matter_ids_truncated"] = len(exported) < len(dark_ids)
    return {
        "schema_version": full["schema_version"],
        "package_version": full["package_version"],
        "summary_only": True,
        "query_ids": full["query_ids"],
        "corpus_fingerprint": full["corpus_fingerprint"],
        "workload_fingerprint": full["workload_fingerprint"],
        "metadata": full["metadata"],
        "report": report,
    }


def save_probe_summary(result: ProbeResult, path: str, *, max_exported_dark_ids: int = 1000) -> None:
    with _open_text(path, "w") as file:
        json.dump(
            probe_summary_to_json(result, max_exported_dark_ids=max_exported_dark_ids),
            file,
            ensure_ascii=False,
            indent=2,
        )
