"""test_qrels.py — qrels 'lost gold' cross-check (first-class feature)."""

from __future__ import annotations

import json
import os
import tempfile

from retrieval_fairness.qrels import QrelsValidation, validate_qrels


def _write(d, name, obj):
    path = os.path.join(d, name)
    if name.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for qid in obj:
                f.write(json.dumps({"id": qid}) + "\n")
    return path


def test_lost_gold_and_recall():
    """dark_matter ∩ relevant = lost gold; recall@k over qrels pairs."""
    with tempfile.TemporaryDirectory() as d:
        # corpus A,B,C,D; q1 hits A,B; q2 hits A. C,D = dark matter.
        # qrels: q1 relevant to C (lost gold); q2 relevant to A (found).
        probe = {
            "freqs": {"A": 2, "B": 1, "C": 0, "D": 0},
            "hits_per_query": [["A", "B"], ["A"]],
            "report": {},
        }
        qrels = {"q1": {"C": 1}, "q2": {"A": 1}}
        pp = _write(d, "probe.json", probe)
        qp = _write(d, "qrels.json", qrels)
        qq = _write(d, "queries.jsonl", ["q1", "q2"])

        res = validate_qrels(pp, qp, qq)

    assert isinstance(res, QrelsValidation)
    assert res.dark_matter == 2
    assert res.relevant_in_corpus == 2  # A and C
    assert res.dark_and_relevant == 1  # C
    assert res.dark_relevant_ids == ["C"]
    assert res.dark_relevant_pct_of_relevant == 0.5
    assert res.qrels_pairs_total == 2
    assert res.qrels_pairs_in_topk == 1  # A found, C not
    assert res.recall_at_k == 0.5


def test_to_dict_roundtrips():
    v = QrelsValidation(
        n_chunks=10,
        n_queries=5,
        dark_matter=4,
        relevant_in_corpus=3,
        dark_and_relevant=2,
        dark_relevant_pct_of_dark=0.5,
        dark_relevant_pct_of_relevant=0.67,
        qrels_pairs_total=3,
        qrels_pairs_in_topk=1,
        micro_recall_at_k=round(1 / 3, 4),
        dark_relevant_ids=["x"],
    )
    d = v.to_dict()
    assert d["dark_and_relevant"] == 2
    assert d["recall_at_k"] == round(1 / 3, 4)
    assert d["dark_relevant_ids"] == ["x"]


def test_mismatched_queries_raises():
    import pytest

    with tempfile.TemporaryDirectory() as d:
        probe = {"freqs": {"A": 1}, "hits_per_query": [["A"]], "report": {}}
        pp = _write(d, "probe.json", probe)
        qp = _write(d, "qrels.json", {})
        qq = _write(d, "queries.jsonl", ["q1", "q2"])  # 2 queries vs 1 hit
        with pytest.raises(ValueError, match="does not match"):
            validate_qrels(pp, qp, qq)


def test_empty_qrels_safe():
    with tempfile.TemporaryDirectory() as d:
        probe = {"freqs": {"A": 1, "B": 0}, "hits_per_query": [["A"]], "report": {}}
        pp = _write(d, "probe.json", probe)
        qp = _write(d, "qrels.json", {})
        qq = _write(d, "queries.jsonl", ["q1"])
        res = validate_qrels(pp, qp, qq)
    assert res.dark_matter == 1
    assert res.relevant_in_corpus == 0
    assert res.dark_and_relevant == 0
    assert res.recall_at_k == 0.0  # no pairs -> 0 by definition
