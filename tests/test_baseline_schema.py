from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval_fairness.adapters.inmemory import InMemoryVectorStore
from retrieval_fairness.probe import probe
from retrieval_fairness.serialize import (
    load_probe,
    probe_summary_to_json,
    save_probe,
)
from retrieval_fairness.types import Chunk, Query

FIXTURES = Path(__file__).parent / "fixtures"


def _result():
    chunks = [
        Chunk("A", "a", [1.0, 0.0]),
        Chunk("B", "b", [0.0, 1.0]),
        Chunk("C", "c", [-1.0, 0.0]),
    ]
    queries = [Query("q1", [1.0, 0.0]), Query("q2", [0.0, 1.0])]
    return probe(InMemoryVectorStore(chunks), queries, top_k=1)


def _saved_dict(tmp_path: Path) -> tuple[Path, dict]:
    path = tmp_path / "probe.json"
    save_probe(_result(), str(path))
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_v3_roundtrip_identities_and_metadata(tmp_path):
    path, raw = _saved_dict(tmp_path)
    loaded = load_probe(str(path))
    assert raw["schema_version"] == 3
    assert raw["package_version"] == "0.2.0"
    assert raw["metadata"]["created_at"].endswith("Z")
    assert loaded.query_ids == ["q1", "q2"]
    assert loaded.corpus_fingerprint.startswith("sha256:")
    assert loaded.workload_fingerprint.startswith("sha256:")
    assert loaded.workload_ids_fingerprint.startswith("sha256:")
    assert loaded.workload_content_fingerprint is None
    assert loaded.corpus_set_fingerprint.startswith("sha256:")
    assert loaded.corpus_content_fingerprint.startswith("sha256:")


@pytest.mark.parametrize(
    ("field", "value"),
    [("coverage_pct", 0.99), ("gini", 0.99), ("dark_matter_pct", 0.01)],
)
def test_tampered_report_is_rejected(tmp_path, field, value):
    path, raw = _saved_dict(tmp_path)
    raw["report"][field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="saved report"):
        load_probe(str(path))
    relaxed = load_probe(str(path), strict_integrity=False)
    assert relaxed.report.coverage_pct == 2 / 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["freqs"].update(A=-1),
        lambda data: data["hits_per_query"][0].append("A"),
        lambda data: data["hits_per_query"][0].append("UNKNOWN"),
        lambda data: data["query_ids"].append("q1"),
        lambda data: data.update(corpus_fingerprint="sha256:bad"),
    ],
)
def test_corrupt_raw_data_is_rejected(tmp_path, mutation):
    path, raw = _saved_dict(tmp_path)
    mutation(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_probe(str(path))


def test_legacy_fixture_rebuilt_and_marked():
    result = load_probe(str(FIXTURES / "baseline_v0_1_0.json"))
    assert result.schema_version == 1
    assert result.metadata["legacy_schema"] is True
    assert result.metadata["legacy_positional_alignment"] is True
    assert result.report.coverage_pct == 2 / 3


def test_legacy_inconsistent_report_strict_vs_relaxed():
    path = str(FIXTURES / "baseline_inconsistent_report.json")
    with pytest.raises(ValueError, match="saved report"):
        load_probe(path)
    assert load_probe(path, strict_integrity=False).report.coverage_pct == 0.5


def test_compact_summary_has_exact_counts_and_bounded_lorenz():
    summary = probe_summary_to_json(_result(), max_exported_dark_ids=0)
    assert summary["summary_only"] is True
    assert "freqs" not in summary
    assert "hits_per_query" not in summary
    assert "query_ids" not in summary
    assert summary["report"]["dark_matter_count"] == 1
    assert "dark_matter_ids" not in summary["report"]
    assert summary["report"]["lorenz_points_exported"] <= 512
