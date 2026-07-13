from __future__ import annotations

import json

import pytest

from retrieval_fairness.adapters.inmemory import InMemoryVectorStore
from retrieval_fairness.cli import main
from retrieval_fairness.dashboard import build_html
from retrieval_fairness.diff import diff_reports, per_query_overlap
from retrieval_fairness.probe import ProbeResult, probe
from retrieval_fairness.qrels import validate_qrels
from retrieval_fairness.serialize import save_probe
from retrieval_fairness.types import Chunk, Query
from retrieval_fairness.validation import validate_vector


def _probe(query_order=("q1", "q2"), chunks=None):
    chunks = chunks or [
        Chunk("A", "a", [1.0, 0.0]),
        Chunk("B", "b", [0.0, 1.0]),
    ]
    vectors = {"q1": [1.0, 0.0], "q2": [0.0, 1.0]}
    queries = [Query(qid, vectors[qid]) for qid in query_order]
    return probe(InMemoryVectorStore(chunks), queries, top_k=1)


def test_overlap_aligns_reordered_workload():
    baseline = _probe()
    candidate = _probe(("q2", "q1"))
    assert diff_reports(baseline, candidate).mean_query_overlap == 1.0


def test_overlap_rejects_same_count_different_ids():
    with pytest.raises(ValueError, match="query ID sets differ"):
        per_query_overlap([["A"]], [["A"]], ["q1"], ["different"])


def test_corpus_policy_same_and_allow_change():
    baseline = _probe()
    candidate = _probe(chunks=[Chunk("A", "a", [1.0, 0.0]), Chunk("C", "c", [0.0, 1.0])])
    with pytest.raises(ValueError, match="corpus fingerprints"):
        diff_reports(baseline, candidate)
    report = diff_reports(baseline, candidate, corpus_policy="allow-change")
    assert report.corpus_changed is True
    assert report.n_chunks_delta == 0
    assert "denominators" in str(report)


def test_qrels_v2_without_queries_and_positive_grades(tmp_path):
    result = _probe()
    probe_path = tmp_path / "probe.json"
    qrels_path = tmp_path / "qrels.json"
    save_probe(result, str(probe_path))
    qrels_path.write_text(
        json.dumps({"q1": {"A": 0, "B": 1}, "q2": {"B": 2, "A": -1}}),
        encoding="utf-8",
    )
    validation = validate_qrels(str(probe_path), str(qrels_path))
    assert validation.qrels_pairs_total == 2
    assert validation.micro_recall_at_k == 0.5
    assert validation.macro_recall_at_k == 0.5
    assert validation.queries_with_relevant_docs == 2


def test_qrels_macro_ignores_query_without_relevant_docs(tmp_path):
    result = _probe()
    probe_path = tmp_path / "probe.json"
    qrels_path = tmp_path / "qrels.json"
    save_probe(result, str(probe_path))
    qrels_path.write_text(json.dumps({"q1": {"A": 2}}), encoding="utf-8")
    validation = validate_qrels(str(probe_path), str(qrels_path), min_relevance_grade=2)
    assert validation.micro_recall_at_k == 1.0
    assert validation.macro_recall_at_k == 1.0
    assert validation.per_query_recall == {"q1": 1.0}


@pytest.mark.parametrize(
    ("store", "args", "option"),
    [
        ("inmemory", [], "--corpus"),
        ("faiss", [], "--index-path"),
        ("pgvector", [], "--database-url"),
        ("qdrant", [], "--url"),
        ("qdrant", ["--url", "http://localhost"], "--collection"),
    ],
)
def test_cli_store_required_args_exit_two(tmp_path, capsys, store, args, option):
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"id":"q","vector":[1.0]}\n', encoding="utf-8")
    code = main(["probe", "--store", store, "--queries", str(queries), *args])
    captured = capsys.readouterr()
    assert code == 2
    assert option in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")]])
def test_vector_validation_rejects_empty_and_nonfinite(bad):
    with pytest.raises(ValueError):
        validate_vector(bad, name="vector")


def test_inmemory_rejects_duplicate_ids_and_dimension_mismatch():
    with pytest.raises(ValueError, match="duplicate"):
        InMemoryVectorStore([Chunk("A", "", [1.0]), Chunk("A", "", [2.0])])
    with pytest.raises(ValueError, match="dimension"):
        InMemoryVectorStore([Chunk("A", "", [1.0]), Chunk("B", "", [1.0, 2.0])])


def test_dashboard_small_ragged_nan_and_sampling():
    result = _probe()
    one = build_html(result, chunks_vectors=[[1.0, 2.0]], chunk_ids=["A"])
    assert "минимум 2" in one
    with pytest.raises(ValueError, match="dimension"):
        build_html(result, chunks_vectors=[[1.0, 2.0], [1.0]], chunk_ids=["A", "B"])
    with pytest.raises(ValueError, match="finite"):
        build_html(result, chunks_vectors=[[1.0, 2.0], [1.0, float("nan")]], chunk_ids=["A", "B"])
    vectors = [[float(i), float(i % 7)] for i in range(50)]
    ids = [f"C{i}" for i in range(50)]
    sampled = build_html(result, chunks_vectors=vectors, chunk_ids=ids, max_pca_points=10)
    assert "displayed 10 of 50 chunks" in sampled


def test_probe_requires_unique_query_ids_and_positive_top_k():
    store = InMemoryVectorStore([Chunk("A", "", [1.0])])
    with pytest.raises(ValueError, match="duplicate"):
        probe(store, [Query("q", [1.0]), Query("q", [1.0])])
    with pytest.raises(ValueError, match="positive"):
        probe(store, [Query("q", [1.0])], top_k=0)


def test_serialize_rejects_reportless_result(tmp_path):
    with pytest.raises(ValueError, match="report"):
        save_probe(ProbeResult(freqs={}), str(tmp_path / "x.json"))
