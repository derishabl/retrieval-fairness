"""Versioned, integrity-checked serialization for :class:`ProbeResult`."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO, cast

from retrieval_fairness import __version__
from retrieval_fairness.metrics import FairnessReport, build_report, downsample_lorenz, retrieval_frequencies
from retrieval_fairness.probe import (
    SCHEMA_VERSION,
    ProbeResult,
    stable_ids_fingerprint,
    stable_set_fingerprint,
)
from retrieval_fairness.provenance import sanitize_metadata
from retrieval_fairness.validation import require_non_negative_int, validate_unique_ids

_REPORT_METRICS = (
    "coverage_pct",
    "dark_matter_pct",
    "gini",
    "hub_capture_top5",
    "hub_capture_top10",
)
_SUPPORTED_SCHEMAS = {1, 2, SCHEMA_VERSION}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _open_text(path: str, mode: str) -> TextIO:
    if str(path).endswith(".gz"):
        return cast(TextIO, gzip.open(path, mode + "t", encoding="utf-8"))
    return cast(TextIO, open(path, mode, encoding="utf-8"))


def _validate_raw(freqs: dict[str, int], hits: list[list[str]], query_ids: list[str], top_k: int) -> None:
    validate_unique_ids(list(freqs), name="corpus IDs")
    validate_unique_ids(query_ids, name="query_ids")
    if len(query_ids) != len(hits):
        raise ValueError(f"len(query_ids)={len(query_ids)} != len(hits_per_query)={len(hits)}")
    for chunk_id, value in freqs.items():
        require_non_negative_int(value, f"frequency[{chunk_id!r}]")
    for index, row in enumerate(hits):
        if not isinstance(row, list) or any(not isinstance(chunk_id, str) for chunk_id in row):
            raise ValueError(f"hits_per_query[{index}] must be a list of string IDs")
        validate_unique_ids(row, name=f"hits_per_query[{index}]")
        if len(row) > top_k:
            raise ValueError(f"hits_per_query[{index}] has {len(row)} hits, exceeding top_k={top_k}")
    rebuilt = retrieval_frequencies(hits, list(freqs))
    if rebuilt != freqs:
        changed = [chunk_id for chunk_id in freqs if rebuilt[chunk_id] != freqs[chunk_id]][:5]
        raise ValueError(f"frequencies do not match hits_per_query; examples: {changed!r}")


def _optional_fingerprint(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{name} must be a sha256: fingerprint")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256: fingerprint") from exc
    return value


def _optional_revision(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when provided")
    return value


def _serialization_header(result: ProbeResult) -> dict[str, Any]:
    if result.report is None:
        raise ValueError("ProbeResult.report is required for serialization")
    query_ids = list(result.query_ids)
    if not query_ids and result.hits_per_query:
        raise ValueError("schema v3 serialization requires query_ids")
    _validate_raw(result.freqs, result.hits_per_query, query_ids, result.report.top_k)

    expected_corpus_order = stable_ids_fingerprint(list(result.freqs))
    expected_corpus_set = stable_set_fingerprint(list(result.freqs))
    expected_workload_order = stable_ids_fingerprint(query_ids)
    expected_workload_set = stable_set_fingerprint(query_ids)

    corpus_order = result.corpus_order_fingerprint or result.corpus_fingerprint or expected_corpus_order
    corpus_set = result.corpus_set_fingerprint or expected_corpus_set
    workload_ids = result.workload_ids_fingerprint or expected_workload_set
    if corpus_order != expected_corpus_order:
        raise ValueError("ProbeResult corpus_order_fingerprint does not match raw corpus IDs")
    if corpus_set != expected_corpus_set:
        raise ValueError("ProbeResult corpus_set_fingerprint does not match raw corpus IDs")
    if workload_ids != expected_workload_set:
        raise ValueError("ProbeResult workload_ids_fingerprint does not match raw query IDs")
    if result.workload_fingerprint and result.workload_fingerprint != expected_workload_order:
        raise ValueError("ProbeResult workload_fingerprint does not match raw query IDs")

    metadata = sanitize_metadata(dict(result.metadata))
    metadata.setdefault("created_at", _utc_now())
    metadata.setdefault("python_version", None)
    metadata.setdefault("platform", None)
    metadata.setdefault("adapter", "unknown")
    metadata.setdefault("embedder", None)
    metadata["top_k"] = result.report.top_k
    metadata["corpus_revision"] = result.corpus_revision or metadata.get("corpus_revision")
    metadata["workload_revision"] = result.workload_revision or metadata.get("workload_revision")

    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "query_ids": query_ids,
        # Compatibility aliases retain v2 ordered-ID semantics.
        "corpus_fingerprint": expected_corpus_order,
        "workload_fingerprint": expected_workload_order,
        "workload_ids_fingerprint": workload_ids,
        "workload_content_fingerprint": _optional_fingerprint(
            result.workload_content_fingerprint, "workload_content_fingerprint"
        ),
        "workload_revision": _optional_revision(
            result.workload_revision or metadata.get("workload_revision"), "workload_revision"
        ),
        "corpus_set_fingerprint": corpus_set,
        "corpus_order_fingerprint": corpus_order,
        "corpus_content_fingerprint": _optional_fingerprint(
            result.corpus_content_fingerprint, "corpus_content_fingerprint"
        ),
        "corpus_revision": _optional_revision(
            result.corpus_revision or metadata.get("corpus_revision"), "corpus_revision"
        ),
        "index_mapping_fingerprint": _optional_fingerprint(
            result.index_mapping_fingerprint, "index_mapping_fingerprint"
        ),
        "metadata": metadata,
    }


def probe_to_json(result: ProbeResult) -> dict[str, Any]:
    """Convert a valid result to the full baseline schema."""
    header = _serialization_header(result)
    report = result.report
    if report is None:  # narrowed for static type checkers; header already validates this
        raise ValueError("ProbeResult.report is required for serialization")
    header.update(
        {
            "freqs": result.freqs,
            "hits_per_query": result.hits_per_query,
            "report": report.to_dict(),
        }
    )
    return header


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
    output: dict[str, int] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("frequency IDs must be non-empty strings")
        if strict and (isinstance(raw, bool) or not isinstance(raw, int)):
            raise ValueError(f"frequency[{key!r}] must be an integer")
        try:
            parsed = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frequency[{key!r}] must be an integer") from exc
        require_non_negative_int(parsed, f"frequency[{key!r}]")
        output[key] = parsed
    return output


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
    """Load a full baseline and always rebuild metrics from raw observations."""
    with _open_text(path, "r") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("baseline root must be a JSON object")
    if data.get("summary_only") is True:
        raise ValueError("summary-only artifacts cannot be loaded as raw probe baselines")

    schema_version = data.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("schema_version must be an integer")
    if schema_version not in _SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported baseline schema_version={schema_version}")
    legacy = schema_version == 1

    freqs = _parse_freqs(data.get("freqs"), strict=strict_integrity)
    raw_hits = data.get("hits_per_query", [])
    if not isinstance(raw_hits, list):
        raise ValueError("hits_per_query must be a list")
    hits: list[list[str]] = []
    for index, row in enumerate(raw_hits):
        if not isinstance(row, list) or any(not isinstance(chunk_id, str) for chunk_id in row):
            raise ValueError(f"hits_per_query[{index}] must be a list of string IDs")
        hits.append(list(row))

    raw_query_ids = data.get("query_ids", [])
    if not isinstance(raw_query_ids, list) or any(
        not isinstance(query_id, str) for query_id in raw_query_ids
    ):
        raise ValueError("query_ids must be a list of strings")
    query_ids = list(raw_query_ids)
    if schema_version >= 2 and "query_ids" not in data:
        raise ValueError(f"schema v{schema_version} requires query_ids")

    report_data = data.get("report", {})
    if not isinstance(report_data, dict):
        raise ValueError("report must be an object")
    raw_metadata = data.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("metadata must be an object")
    metadata = sanitize_metadata(raw_metadata)
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
        effective_ids = query_ids if query_ids else [f"legacy:{index}" for index in range(len(hits))]
        _validate_raw(freqs, hits, effective_ids, top_k)
    elif any(len(row) > top_k for row in hits):
        metadata["legacy_integrity_relaxed"] = True

    n_queries = len(query_ids) if query_ids else len(hits)
    rebuilt_report = build_report(freqs, n_queries=n_queries, top_k=top_k)
    if strict_integrity and report_data and not _report_matches(report_data, rebuilt_report):
        raise ValueError("saved report does not match metrics rebuilt from raw data")

    corpus_fingerprint = data.get("corpus_fingerprint")
    workload_fingerprint = data.get("workload_fingerprint")
    corpus_set_fingerprint: str | None = None
    corpus_order_fingerprint: str | None = None
    workload_ids_fingerprint: str | None = None
    if schema_version >= 2:
        expected_corpus_order = stable_ids_fingerprint(list(freqs))
        expected_workload_order = stable_ids_fingerprint(query_ids)
        if corpus_fingerprint != expected_corpus_order:
            raise ValueError("corpus_fingerprint does not match corpus IDs")
        if workload_fingerprint != expected_workload_order:
            raise ValueError("workload_fingerprint does not match query IDs")
        corpus_set_fingerprint = stable_set_fingerprint(list(freqs))
        corpus_order_fingerprint = expected_corpus_order
        workload_ids_fingerprint = stable_set_fingerprint(query_ids)

    if schema_version == SCHEMA_VERSION:
        corpus_set_fingerprint = _optional_fingerprint(
            data.get("corpus_set_fingerprint"), "corpus_set_fingerprint"
        )
        corpus_order_fingerprint = _optional_fingerprint(
            data.get("corpus_order_fingerprint"), "corpus_order_fingerprint"
        )
        workload_ids_fingerprint = _optional_fingerprint(
            data.get("workload_ids_fingerprint"), "workload_ids_fingerprint"
        )
        if corpus_set_fingerprint != stable_set_fingerprint(list(freqs)):
            raise ValueError("corpus_set_fingerprint does not match corpus IDs")
        if corpus_order_fingerprint != stable_ids_fingerprint(list(freqs)):
            raise ValueError("corpus_order_fingerprint does not match corpus IDs")
        if workload_ids_fingerprint != stable_set_fingerprint(query_ids):
            raise ValueError("workload_ids_fingerprint does not match query IDs")

    workload_revision = _optional_revision(
        data.get("workload_revision", metadata.get("workload_revision")), "workload_revision"
    )
    corpus_revision = _optional_revision(
        data.get("corpus_revision", metadata.get("corpus_revision")), "corpus_revision"
    )
    return ProbeResult(
        freqs=freqs,
        hits_per_query=hits,
        query_ids=query_ids,
        report=rebuilt_report,
        schema_version=schema_version,
        corpus_fingerprint=corpus_fingerprint,
        workload_fingerprint=workload_fingerprint,
        workload_ids_fingerprint=workload_ids_fingerprint,
        workload_content_fingerprint=_optional_fingerprint(
            data.get("workload_content_fingerprint"), "workload_content_fingerprint"
        ),
        workload_revision=workload_revision,
        corpus_set_fingerprint=corpus_set_fingerprint,
        corpus_order_fingerprint=corpus_order_fingerprint,
        corpus_content_fingerprint=_optional_fingerprint(
            data.get("corpus_content_fingerprint"), "corpus_content_fingerprint"
        ),
        corpus_revision=corpus_revision,
        index_mapping_fingerprint=_optional_fingerprint(
            data.get("index_mapping_fingerprint"), "index_mapping_fingerprint"
        ),
        metadata=metadata,
    )


def _rounded_points(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[round(x, 6), round(y, 6)] for x, y in points]


def probe_summary_to_json(
    result: ProbeResult,
    *,
    max_exported_dark_ids: int = 0,
    max_lorenz_points: int = 512,
) -> dict[str, Any]:
    """Build a compact, non-raw summary without materializing full JSON."""
    require_non_negative_int(max_exported_dark_ids, "max_exported_dark_ids")
    require_non_negative_int(max_lorenz_points, "max_lorenz_points")
    if max_lorenz_points < 2:
        raise ValueError("max_lorenz_points must be at least 2")
    header = _serialization_header(result)
    report = result.report
    if report is None:
        raise ValueError("ProbeResult.report is required for serialization")
    points = downsample_lorenz(result.freqs, max_lorenz_points)
    report_summary: dict[str, Any] = {
        "n_chunks": report.n_chunks,
        "n_queries": report.n_queries,
        "top_k": report.top_k,
        "coverage_pct": round(report.coverage_pct, 4),
        "dark_matter_pct": round(report.dark_matter_pct, 4),
        "gini": round(report.gini, 4),
        "hub_capture_top5": round(report.hub_capture_top5, 4),
        "hub_capture_top10": round(report.hub_capture_top10, 4),
        "hub_leaderboard": report.hub_leaderboard,
        "dark_matter_count": report.dark_matter_count,
        "reachability_ceiling": report.reachability_ceiling,
        "coverage_of_ceiling": round(report.coverage_of_ceiling, 4),
        "lorenz_curve": _rounded_points(points),
        "lorenz_points_total": report.lorenz_points_total,
        "lorenz_points_exported": len(points),
        "downsampled": len(points) < report.lorenz_points_total,
    }
    if max_exported_dark_ids:
        exported = report.dark_matter_ids[:max_exported_dark_ids]
        report_summary["dark_matter_ids"] = exported
        report_summary["dark_matter_ids_exported"] = len(exported)
        report_summary["dark_matter_ids_truncated"] = len(exported) < report.dark_matter_count

    # Query IDs are intentionally absent: identity hashes and exact counts are
    # enough for a dashboard/CI artifact, while raw comparison uses save_probe.
    header.pop("query_ids", None)
    header["summary_only"] = True
    header["report"] = report_summary
    return header


def save_probe_summary(
    result: ProbeResult,
    path: str,
    *,
    max_exported_dark_ids: int = 0,
    max_lorenz_points: int = 512,
) -> None:
    target = str(path)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with _open_text(target, "w") as file:
        json.dump(
            probe_summary_to_json(
                result,
                max_exported_dark_ids=max_exported_dark_ids,
                max_lorenz_points=max_lorenz_points,
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )
