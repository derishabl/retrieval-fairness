"""PR-friendly migration blast-radius artifacts derived from a probe diff."""

from __future__ import annotations

import csv
import html
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from retrieval_fairness.diff import DiffReport, newly_dark_matter, rescued_from_dark_matter
from retrieval_fairness.probe import ProbeResult
from retrieval_fairness.validation import require_non_negative_int


@dataclass(frozen=True)
class BlastRadiusEntry:
    """One chunk whose zero/non-zero exposure state changed during migration."""

    change: str
    chunk_id: str
    baseline_frequency: int
    candidate_frequency: int
    delta: int
    text: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "change": self.change,
            "chunk_id": self.chunk_id,
            "baseline_frequency": self.baseline_frequency,
            "candidate_frequency": self.candidate_frequency,
            "delta": self.delta,
            "text": self.text,
        }


@dataclass(frozen=True)
class BlastRadiusReport:
    """Actionable per-chunk state changes plus the parent diff summary."""

    new_dark_matter: tuple[BlastRadiusEntry, ...]
    rescued: tuple[BlastRadiusEntry, ...]
    coverage_delta: float
    dark_matter_delta: float
    mean_query_overlap: float
    corpus_changed: bool
    legacy_positional_alignment: bool
    workload_policy: str
    corpus_policy: str

    @property
    def entries(self) -> tuple[BlastRadiusEntry, ...]:
        return self.new_dark_matter + self.rescued

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "new_dark_matter_count": len(self.new_dark_matter),
                "rescued_count": len(self.rescued),
                "coverage_delta": round(self.coverage_delta, 4),
                "dark_matter_delta": round(self.dark_matter_delta, 4),
                "mean_query_overlap": round(self.mean_query_overlap, 4),
                "corpus_changed": self.corpus_changed,
                "legacy_positional_alignment": self.legacy_positional_alignment,
                "workload_policy": self.workload_policy,
                "corpus_policy": self.corpus_policy,
            },
            "new_dark_matter": [entry.to_dict() for entry in self.new_dark_matter],
            "rescued": [entry.to_dict() for entry in self.rescued],
        }


def _validate_chunk_texts(chunk_texts: Mapping[str, str] | None) -> dict[str, str]:
    if chunk_texts is None:
        return {}
    output: dict[str, str] = {}
    for chunk_id, text in chunk_texts.items():
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("chunk_texts keys must be non-empty string IDs")
        if not isinstance(text, str):
            raise ValueError(f"chunk_texts[{chunk_id!r}] must be text")
        output[chunk_id] = text
    return output


def build_blast_radius(
    diff: DiffReport,
    baseline: ProbeResult,
    candidate: ProbeResult,
    *,
    chunk_texts: Mapping[str, str] | None = None,
) -> BlastRadiusReport:
    """Build deterministic newly-dark/rescued rows from trusted raw frequencies."""
    texts = _validate_chunk_texts(chunk_texts)

    def entry(change: str, chunk_id: str) -> BlastRadiusEntry:
        baseline_frequency = baseline.freqs.get(chunk_id, 0)
        candidate_frequency = candidate.freqs.get(chunk_id, 0)
        return BlastRadiusEntry(
            change=change,
            chunk_id=chunk_id,
            baseline_frequency=baseline_frequency,
            candidate_frequency=candidate_frequency,
            delta=candidate_frequency - baseline_frequency,
            text=texts.get(chunk_id),
        )

    newly_dark = [
        entry("new_dark_matter", chunk_id) for chunk_id in newly_dark_matter(baseline.freqs, candidate.freqs)
    ]
    rescued = [
        entry("rescued", chunk_id) for chunk_id in rescued_from_dark_matter(baseline.freqs, candidate.freqs)
    ]
    newly_dark.sort(key=lambda item: (-item.baseline_frequency, item.chunk_id))
    rescued.sort(key=lambda item: (-item.candidate_frequency, item.chunk_id))

    return BlastRadiusReport(
        new_dark_matter=tuple(newly_dark),
        rescued=tuple(rescued),
        coverage_delta=diff.coverage_delta,
        dark_matter_delta=diff.dark_matter_delta,
        mean_query_overlap=diff.mean_query_overlap,
        corpus_changed=diff.corpus_changed,
        legacy_positional_alignment=diff.legacy_positional_alignment,
        workload_policy=diff.workload_policy,
        corpus_policy=diff.corpus_policy,
    )


def _markdown_cell(value: object) -> str:
    normalized = str(value).replace("\r\n", "\n").replace("\r", "\n")
    escaped = html.escape(normalized, quote=False)
    for marker in ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "!", "|"):
        escaped = escaped.replace(marker, f"\\{marker}")
    return escaped.replace("\n", "<br>")


def _truncate_text(text: str, max_text_chars: int) -> str:
    if max_text_chars == 0 or len(text) <= max_text_chars:
        return text
    if max_text_chars == 1:
        return "…"
    return text[: max_text_chars - 1].rstrip() + "…"


def _markdown_section(
    heading: str,
    description: str,
    entries: tuple[BlastRadiusEntry, ...],
    *,
    include_text: bool,
    max_text_chars: int,
) -> list[str]:
    lines = [f"## {heading} ({len(entries)})", "", description, ""]
    if not entries:
        lines.extend(["_No chunks in this category._", ""])
        return lines

    columns = ["Chunk ID", "Baseline frequency", "Candidate frequency", "Delta"]
    if include_text:
        columns.append("Text")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for entry in entries:
        row = [
            _markdown_cell(entry.chunk_id),
            str(entry.baseline_frequency),
            str(entry.candidate_frequency),
            f"{entry.delta:+d}",
        ]
        if include_text:
            row.append(_markdown_cell(_truncate_text(entry.text or "", max_text_chars)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def render_blast_radius_markdown(
    report: BlastRadiusReport,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    max_text_chars: int = 240,
) -> str:
    """Render a deterministic GitHub-friendly Markdown report.

    Chunk text is bounded to keep large migration inventories reviewable. Set
    ``max_text_chars=0`` for lossless Markdown; CSV output is always lossless.
    """
    max_text_chars = require_non_negative_int(max_text_chars, "max_text_chars")
    entries = report.entries
    include_text = any(entry.text is not None for entry in entries)
    text_truncated = bool(max_text_chars) and any(
        entry.text is not None and len(entry.text) > max_text_chars for entry in entries
    )
    lines = [
        "# Migration blast radius",
        "",
        f"**Baseline:** {_markdown_cell(baseline_label)} → **Candidate:** {_markdown_cell(candidate_label)}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Newly dark-matter chunks | {len(report.new_dark_matter)} |",
        f"| Rescued chunks | {len(report.rescued)} |",
        f"| Coverage delta | {report.coverage_delta * 100:+.2f} pp |",
        f"| Dark-matter delta | {report.dark_matter_delta * 100:+.2f} pp |",
        f"| Mean per-query overlap | {report.mean_query_overlap:.4f} |",
        f"| Workload policy | {_markdown_cell(report.workload_policy)} |",
        f"| Corpus policy | {_markdown_cell(report.corpus_policy)} |",
        "",
    ]
    if report.corpus_changed:
        lines.extend(
            [
                "> **Warning:** corpora differ; coverage denominators and chunk sets are not identical.",
                "",
            ]
        )
    if report.legacy_positional_alignment:
        lines.extend(
            [
                "> **Warning:** legacy positional query alignment was used.",
                "",
            ]
        )
    if text_truncated:
        lines.extend(
            [
                f"> Chunk text is truncated to {max_text_chars} characters per row; CSV output is lossless.",
                "",
            ]
        )
    lines.extend(
        _markdown_section(
            "Newly dark matter",
            "These chunks had exposure in the baseline and zero exposure in the candidate.",
            report.new_dark_matter,
            include_text=include_text,
            max_text_chars=max_text_chars,
        )
    )
    lines.extend(
        _markdown_section(
            "Rescued from dark matter",
            "These chunks had zero exposure in the baseline and gained exposure in the candidate.",
            report.rescued,
            include_text=include_text,
            max_text_chars=max_text_chars,
        )
    )
    lines.append("_Generated by `retrieval-fairness diff`._")
    return "\n".join(lines) + "\n"


def render_blast_radius_csv(report: BlastRadiusReport) -> str:
    """Render a stable machine-readable CSV containing every changed chunk."""
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "change",
            "chunk_id",
            "baseline_frequency",
            "candidate_frequency",
            "delta",
            "text",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for entry in report.entries:
        writer.writerow(entry.to_dict())
    return output.getvalue()


def save_blast_radius(
    report: BlastRadiusReport,
    path: str | Path,
    *,
    output_format: str = "md",
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    max_text_chars: int = 240,
) -> None:
    """Write a Markdown (default) or CSV migration artifact."""
    if output_format not in {"md", "csv"}:
        raise ValueError("blast-radius output_format must be 'md' or 'csv'")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (
        render_blast_radius_markdown(
            report,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            max_text_chars=max_text_chars,
        )
        if output_format == "md"
        else render_blast_radius_csv(report)
    )
    with target.open("w", encoding="utf-8", newline="") as file:
        file.write(content)
