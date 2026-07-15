from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from retrieval_fairness.adapters.inmemory import InMemoryVectorStore
from retrieval_fairness.diff import diff_reports
from retrieval_fairness.metrics import build_report, reachability_ceiling
from retrieval_fairness.probe import ProbeResult, probe
from retrieval_fairness.qrels import QrelsValidation, load_qrels
from retrieval_fairness.serialize import load_probe, probe_summary_to_json, probe_to_json
from retrieval_fairness.types import Chunk, Query
from retrieval_fairness.validation import validate_vector

FIXTURES = Path(__file__).parent / "fixtures"


def _chunks(order: tuple[str, ...] = ("A", "B"), *, changed_text: bool = False) -> list[Chunk]:
    values = {
        "A": Chunk("A", "alpha changed" if changed_text else "alpha", [1.0, 0.0]),
        "B": Chunk("B", "beta", [0.0, 1.0]),
    }
    return [values[chunk_id] for chunk_id in order]


def _queries(
    order: tuple[str, ...] = ("q1", "q2"),
    *,
    changed_text: bool = False,
    include_text: bool = True,
    swap_vectors: bool = False,
) -> list[Query]:
    vectors = {"q1": [1.0, 0.0], "q2": [0.0, 1.0]}
    if swap_vectors:
        vectors = {"q1": [0.0, 1.0], "q2": [1.0, 0.0]}
    texts = {"q1": "alpha question", "q2": "beta question"}
    if changed_text:
        texts["q1"] = "different meaning"
    return [Query(query_id, vectors[query_id], texts[query_id] if include_text else "") for query_id in order]


def _probe(
    *,
    chunks: list[Chunk] | None = None,
    queries: list[Query] | None = None,
    workload_revision: str | None = None,
    corpus_revision: str | None = None,
) -> ProbeResult:
    return probe(
        InMemoryVectorStore(chunks or _chunks()),
        queries or _queries(),
        top_k=1,
        workload_revision=workload_revision,
        corpus_revision=corpus_revision,
    )


def test_same_content_identity_allows_reorder_and_embedder_vector_change():
    baseline = _probe()
    reordered = _probe(chunks=_chunks(("B", "A")), queries=_queries(("q2", "q1")))
    changed_vectors = _probe(queries=_queries(swap_vectors=True))

    reordered_diff = diff_reports(
        baseline,
        reordered,
        corpus_policy="same-content",
        workload_policy="same-content",
    )
    assert reordered_diff.mean_query_overlap == 1.0
    assert reordered_diff.corpus_changed is False
    diff_reports(
        baseline,
        changed_vectors,
        corpus_policy="same-content",
        workload_policy="same-content",
    )


def test_same_content_identity_rejects_changed_query_or_chunk_text():
    baseline = _probe()
    with pytest.raises(ValueError, match="workload content"):
        diff_reports(
            baseline,
            _probe(queries=_queries(changed_text=True)),
            corpus_policy="same-content",
            workload_policy="same-content",
        )
    with pytest.raises(ValueError, match="corpus content"):
        diff_reports(
            baseline,
            _probe(chunks=_chunks(changed_text=True)),
            corpus_policy="same-content",
            workload_policy="same-content",
        )


def test_precomputed_workload_requires_revision_or_same_ids_opt_in():
    without_text = _probe(queries=_queries(include_text=False))
    with pytest.raises(ValueError, match="requires content fingerprints"):
        diff_reports(
            without_text,
            without_text,
            corpus_policy="same-content",
            workload_policy="same-content",
        )
    diff_reports(
        without_text,
        without_text,
        corpus_policy="same-content",
        workload_policy="same-ids",
    )
    revised = _probe(queries=_queries(include_text=False), workload_revision="queries-2026-07")
    diff_reports(
        revised,
        revised,
        corpus_policy="same-content",
        workload_policy="same-content",
    )


def test_v2_baseline_remains_readable():
    result = load_probe(str(FIXTURES / "baseline_v0_1_1_schema_v2.json"))
    assert result.schema_version == 2
    assert result.workload_ids_fingerprint is not None
    assert result.workload_content_fingerprint is None
    assert result.corpus_set_fingerprint is not None


def test_provenance_is_automatic_and_rejects_credentials():
    result = _probe()
    artifact = probe_to_json(result)
    metadata = artifact["metadata"]
    assert metadata["adapter"] == "inmemory"
    assert metadata["distance_metric"] == "cosine"
    assert metadata["python_version"]
    assert "secret" not in json.dumps(metadata).casefold()

    result.metadata["api_key"] = "do-not-persist"
    with pytest.raises(ValueError, match="credentials"):
        probe_to_json(result)


def test_summary_is_bounded_non_raw_and_rejected_by_loader(tmp_path):
    count = 100_000
    freqs = {f"chunk-{index}": 0 for index in range(count)}
    result = ProbeResult(
        freqs=freqs,
        report=build_report(freqs, n_queries=0, top_k=1, detail="summary"),
    )
    summary = probe_summary_to_json(result)
    encoded = json.dumps(summary, separators=(",", ":")).encode()
    assert len(encoded) < 1_000_000
    assert summary["report"]["lorenz_points_exported"] <= 512
    assert summary["report"]["lorenz_points_total"] == count + 1
    assert summary["report"]["downsampled"] is True
    assert "query_ids" not in summary
    assert "dark_matter_ids" not in summary["report"]

    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="summary-only"):
        load_probe(str(path))


def test_build_report_uses_one_frequency_context(monkeypatch):
    import retrieval_fairness.metrics as metrics

    calls = 0
    original = metrics.validate_unique_ids

    def counted(ids, *, name):
        nonlocal calls
        calls += 1
        return original(ids, name=name)

    monkeypatch.setattr(metrics, "validate_unique_ids", counted)
    report = metrics.build_report({"A": 3, "B": 0, "C": 1}, n_queries=2, top_k=2)
    assert report.gini > 0
    assert calls == 1


def test_inmemory_ties_are_deterministic_by_chunk_id():
    chunks = [Chunk(f"chunk-{index:02d}", "same", [1.0, 0.0]) for index in reversed(range(20))]
    store = InMemoryVectorStore(chunks)
    expected = [f"chunk-{index:02d}" for index in range(5)]
    for _ in range(5):
        assert [hit.chunk_id for hit in store.search([1.0, 0.0], 5)] == expected


def test_vector_validation_accepts_ndarray_but_rejects_bool_and_text():
    validate_vector(np.array([1.0, 2.0]), name="vector")
    with pytest.raises(ValueError):
        validate_vector([True, 1.0], name="vector")
    with pytest.raises(ValueError):
        validate_vector(["1.0", 2.0], name="vector")


def test_qrels_rejects_lossy_and_boolean_grades(tmp_path):
    path = tmp_path / "qrels.json"
    for grade in (1.9, True, "1"):
        path.write_text(json.dumps({"q": {"doc": grade}}), encoding="utf-8")
        with pytest.raises(ValueError, match="integral"):
            load_qrels(str(path))
    path.write_text(json.dumps({"q": {"doc": 1.0}}), encoding="utf-8")
    assert load_qrels(str(path)) == {"q": {"doc": 1}}


def test_public_count_validation_and_read_only_recall_alias():
    with pytest.raises(ValueError):
        reachability_ceiling(-1, 1, 1)
    with pytest.raises(ValueError):
        reachability_ceiling(1, -1, 1)
    with pytest.raises(ValueError):
        reachability_ceiling(1, 1, 0)

    result = QrelsValidation(
        n_chunks=1,
        n_queries=1,
        dark_matter=0,
        relevant_in_corpus=1,
        dark_and_relevant=0,
        dark_relevant_pct_of_dark=0.0,
        dark_relevant_pct_of_relevant=0.0,
        qrels_pairs_total=1,
        qrels_pairs_in_topk=1,
        micro_recall_at_k=1.0,
    )
    assert result.recall_at_k == 1.0
    with pytest.raises(AttributeError):
        result.recall_at_k = 0.0
    assert result.to_dict()["recall_at_k"] == result.to_dict()["micro_recall_at_k"]
