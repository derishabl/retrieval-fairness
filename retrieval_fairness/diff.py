"""Integrity-aware regression diff between two probe runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval_fairness.probe import ProbeResult
from retrieval_fairness.validation import validate_unique_ids


def per_chunk_delta(baseline: dict[str, int], candidate: dict[str, int]) -> dict[str, int]:
    keys = set(baseline) | set(candidate)
    return {key: candidate.get(key, 0) - baseline.get(key, 0) for key in keys}


def newly_dark_matter(baseline: dict[str, int], candidate: dict[str, int]) -> list[str]:
    return [key for key in baseline if baseline[key] > 0 and candidate.get(key, 0) == 0]


def rescued_from_dark_matter(baseline: dict[str, int], candidate: dict[str, int]) -> list[str]:
    return [key for key in candidate if baseline.get(key, 0) == 0 and candidate[key] > 0]


def _workload_mismatch(baseline_ids: list[str], candidate_ids: list[str]) -> ValueError:
    baseline_set, candidate_set = set(baseline_ids), set(candidate_ids)
    missing = sorted(baseline_set - candidate_set)
    added = sorted(candidate_set - baseline_set)
    return ValueError(
        "query ID sets differ "
        f"(baseline={len(baseline_ids)}, candidate={len(candidate_ids)}, "
        f"missing={len(missing)} {missing[:5]!r}, added={len(added)} {added[:5]!r})"
    )


def per_query_overlap(
    baseline_hits: list[list[str]],
    candidate_hits: list[list[str]],
    baseline_query_ids: list[str] | None = None,
    candidate_query_ids: list[str] | None = None,
) -> list[float]:
    """Return per-query Jaccard overlap, aligned by query ID when available."""
    baseline_query_ids = baseline_query_ids or []
    candidate_query_ids = candidate_query_ids or []
    if bool(baseline_query_ids) != bool(candidate_query_ids):
        raise ValueError("only one probe contains query IDs; workload alignment is unsafe")

    candidate_rows = candidate_hits
    if baseline_query_ids:
        validate_unique_ids(baseline_query_ids, name="baseline query IDs")
        validate_unique_ids(candidate_query_ids, name="candidate query IDs")
        if len(baseline_query_ids) != len(baseline_hits):
            raise ValueError("baseline query IDs count does not match hits_per_query")
        if len(candidate_query_ids) != len(candidate_hits):
            raise ValueError("candidate query IDs count does not match hits_per_query")
        if set(baseline_query_ids) != set(candidate_query_ids):
            raise _workload_mismatch(baseline_query_ids, candidate_query_ids)
        candidate_by_id = dict(zip(candidate_query_ids, candidate_hits))
        candidate_rows = [candidate_by_id[qid] for qid in baseline_query_ids]
    elif len(baseline_hits) != len(candidate_hits):
        raise ValueError(
            "per_query_overlap: query counts differ "
            f"(baseline={len(baseline_hits)}, candidate={len(candidate_hits)}); "
            "legacy positional comparison requires equal counts"
        )

    output: list[float] = []
    for baseline_row, candidate_row in zip(baseline_hits, candidate_rows):
        baseline_set, candidate_set = set(baseline_row), set(candidate_row)
        if not baseline_set and not candidate_set:
            output.append(1.0)
            continue
        union = baseline_set | candidate_set
        output.append(len(baseline_set & candidate_set) / len(union))
    return output


@dataclass
class DiffReport:
    coverage_delta: float
    gini_delta: float
    hub_capture_top5_delta: float
    dark_matter_delta: float
    n_chunks_delta: int = 0
    chunk_deltas: dict[str, int] = field(default_factory=dict)
    new_dark_matter: list[str] = field(default_factory=list)
    rescued: list[str] = field(default_factory=list)
    per_query_overlap: list[float] = field(default_factory=list)
    mean_query_overlap: float = 0.0
    worst_losses: list[tuple[str, int]] = field(default_factory=list)
    worst_gains: list[tuple[str, int]] = field(default_factory=list)
    legacy_positional_alignment: bool = False
    corpus_changed: bool = False
    workload_policy: str = "same-ids"
    corpus_policy: str = "same-ids"

    def __str__(self) -> str:
        lines = [
            "=" * 64,
            "REGRESSION DIFF (baseline -> candidate)",
            "=" * 64,
            f"  Coverage delta:     {self.coverage_delta * 100:+.2f}%",
            f"  Dark matter delta:  {self.dark_matter_delta * 100:+.2f}%",
            f"  Gini delta:         {self.gini_delta:+.3f}",
            f"  Hub-capture top5:   {self.hub_capture_top5_delta * 100:+.2f}%",
            f"  Corpus chunks delta:{self.n_chunks_delta:+d}",
            "-" * 64,
            f"  Mean per-query overlap: {self.mean_query_overlap:.3f}",
            f"  Новых dark-matter:  {len(self.new_dark_matter)} чанков",
            f"  Спасённых из dark:  {len(self.rescued)} чанков",
        ]
        if self.legacy_positional_alignment:
            lines.append("  WARNING: legacy positional query alignment was used")
        if self.corpus_changed:
            lines.append("  WARNING: corpora differ; coverage denominators are different")
        lines.extend(["-" * 64, "  Худшие потери (chunk: delta freq):"])
        for chunk_id, delta in self.worst_losses[:10]:
            lines.append(f"    {chunk_id:30} {delta:+d}")
        lines.append("  Наибольшие улучшения (chunk: delta freq):")
        for chunk_id, delta in self.worst_gains[:10]:
            lines.append(f"    {chunk_id:30} {delta:+d}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "coverage_delta": round(self.coverage_delta, 4),
            "dark_matter_delta": round(self.dark_matter_delta, 4),
            "gini_delta": round(self.gini_delta, 4),
            "hub_capture_top5_delta": round(self.hub_capture_top5_delta, 4),
            "n_chunks_delta": self.n_chunks_delta,
            "mean_query_overlap": round(self.mean_query_overlap, 4),
            "new_dark_matter": self.new_dark_matter,
            "rescued": self.rescued,
            "worst_losses": self.worst_losses[:20],
            "worst_gains": self.worst_gains[:20],
            "legacy_positional_alignment": self.legacy_positional_alignment,
            "corpus_changed": self.corpus_changed,
            "workload_policy": self.workload_policy,
            "corpus_policy": self.corpus_policy,
        }


def _corpus_changed(baseline: ProbeResult, candidate: ProbeResult) -> bool:
    baseline_id = baseline.corpus_set_fingerprint or baseline.corpus_fingerprint
    candidate_id = candidate.corpus_set_fingerprint or candidate.corpus_fingerprint
    if baseline_id and candidate_id:
        return baseline_id != candidate_id
    return set(baseline.freqs) != set(candidate.freqs)


def _require_semantic_identity(
    *,
    label: str,
    baseline_content: str | None,
    candidate_content: str | None,
    baseline_revision: str | None,
    candidate_revision: str | None,
    opt_in_policy: str,
) -> None:
    if baseline_content is not None and candidate_content is not None:
        if baseline_content != candidate_content:
            raise ValueError(f"{label} content fingerprints differ")
        return
    if baseline_revision is not None and candidate_revision is not None:
        if baseline_revision != candidate_revision:
            raise ValueError(f"{label} revisions differ")
        return
    raise ValueError(
        f"same-content {label} comparison requires content fingerprints or matching revisions; "
        f"use {opt_in_policy!r} only as an explicit legacy/precomputed opt-in"
    )


def diff_reports(
    baseline: ProbeResult,
    candidate: ProbeResult,
    *,
    corpus_policy: str = "same",
    workload_policy: str = "same-ids",
) -> DiffReport:
    """Compare results under explicit logical identity policies.

    ``same`` remains a compatibility alias for ``same-ids``. CI gates pass
    ``same-content`` by default.
    """
    valid_corpus = {"same", "same-content", "same-ids", "allow-change"}
    if corpus_policy not in valid_corpus:
        raise ValueError(f"corpus_policy must be one of {sorted(valid_corpus)!r}")
    if workload_policy not in {"same-content", "same-ids"}:
        raise ValueError("workload_policy must be 'same-content' or 'same-ids'")
    if baseline.report is None or candidate.report is None:
        raise ValueError("both ProbeResult objects must contain a report")

    normalized_corpus_policy = "same-ids" if corpus_policy == "same" else corpus_policy
    corpus_changed = _corpus_changed(baseline, candidate)
    if corpus_changed and normalized_corpus_policy != "allow-change":
        raise ValueError(
            "corpus fingerprints differ; use corpus_policy='allow-change' for a chunking migration"
        )
    if normalized_corpus_policy == "same-content":
        _require_semantic_identity(
            label="corpus",
            baseline_content=baseline.corpus_content_fingerprint,
            candidate_content=candidate.corpus_content_fingerprint,
            baseline_revision=baseline.corpus_revision,
            candidate_revision=candidate.corpus_revision,
            opt_in_policy="same-ids",
        )
    if workload_policy == "same-content":
        _require_semantic_identity(
            label="workload",
            baseline_content=baseline.workload_content_fingerprint,
            candidate_content=candidate.workload_content_fingerprint,
            baseline_revision=baseline.workload_revision,
            candidate_revision=candidate.workload_revision,
            opt_in_policy="same-ids",
        )

    overlaps = per_query_overlap(
        baseline.hits_per_query,
        candidate.hits_per_query,
        baseline.query_ids,
        candidate.query_ids,
    )
    deltas = per_chunk_delta(baseline.freqs, candidate.freqs)
    losses = sorted(
        ((key, value) for key, value in deltas.items() if value < 0),
        key=lambda item: item[1],
    )
    gains = sorted(
        ((key, value) for key, value in deltas.items() if value > 0),
        key=lambda item: -item[1],
    )
    return DiffReport(
        coverage_delta=candidate.report.coverage_pct - baseline.report.coverage_pct,
        gini_delta=candidate.report.gini - baseline.report.gini,
        hub_capture_top5_delta=(candidate.report.hub_capture_top5 - baseline.report.hub_capture_top5),
        dark_matter_delta=(candidate.report.dark_matter_pct - baseline.report.dark_matter_pct),
        n_chunks_delta=candidate.report.n_chunks - baseline.report.n_chunks,
        chunk_deltas=deltas,
        new_dark_matter=newly_dark_matter(baseline.freqs, candidate.freqs),
        rescued=rescued_from_dark_matter(baseline.freqs, candidate.freqs),
        per_query_overlap=overlaps,
        mean_query_overlap=sum(overlaps) / len(overlaps) if overlaps else 0.0,
        worst_losses=losses,
        worst_gains=gains,
        legacy_positional_alignment=not bool(baseline.query_ids),
        corpus_changed=corpus_changed,
        workload_policy=workload_policy,
        corpus_policy=normalized_corpus_policy,
    )
