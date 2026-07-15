"""test_diff.py — regression diff tests."""

from __future__ import annotations

from retrieval_fairness.diff import (
    diff_reports,
    newly_dark_matter,
    per_chunk_delta,
    per_query_overlap,
    rescued_from_dark_matter,
)
from retrieval_fairness.metrics import FairnessReport
from retrieval_fairness.probe import ProbeResult


def _mock_result(freqs, hits, cov, dm, gini=0.3, hub5=0.4) -> ProbeResult:
    return ProbeResult(
        freqs=freqs,
        hits_per_query=hits,
        report=FairnessReport(
            n_chunks=len(freqs),
            n_queries=len(hits),
            top_k=5,
            coverage_pct=cov,
            dark_matter_pct=dm,
            gini=gini,
            hub_capture_top5=hub5,
            hub_capture_top10=0.5,
        ),
    )


def test_per_chunk_delta():
    d = per_chunk_delta({"a": 5, "b": 3, "c": 0}, {"a": 2, "b": 3, "c": 1})
    assert d == {"a": -3, "b": 0, "c": 1}


def test_newly_dark_matter():
    base = {"a": 5, "b": 3, "c": 2}
    cand = {"a": 0, "b": 3, "c": 1}
    assert newly_dark_matter(base, cand) == ["a"]


def test_rescued_from_dark_matter():
    base = {"a": 0, "b": 3, "c": 0}
    cand = {"a": 2, "b": 3, "c": 0}
    assert rescued_from_dark_matter(base, cand) == ["a"]


def test_per_query_overlap_identical():
    assert per_query_overlap([["a", "b"]], [["a", "b"]]) == [1.0]


def test_per_query_overlap_disjoint():
    assert per_query_overlap([["a", "b"]], [["c", "d"]]) == [0.0]


def test_per_query_overlap_partial():
    # {a,b} vs {b,c}: overlap {b}=1, union {a,b,c}=3 -> 1/3
    ov = per_query_overlap([["a", "b"]], [["b", "c"]])
    assert abs(ov[0] - 1 / 3) < 1e-9


def test_per_query_overlap_rejects_mismatched_length():
    """Баг #4: разное число запросов раньше молча обрезалось zip'ом -> мусор.
    Теперь явная ошибка (тихо неверный результат хуже падения)."""
    try:
        per_query_overlap([["a"], ["b"], ["c"]], [["a"], ["b"]])
        assert False, "expected ValueError on mismatched query count"
    except ValueError:
        pass


def test_diff_reports_propagates_query_count_mismatch():
    base = _mock_result({"a": 1}, [["a"]], cov=1.0, dm=0.0)
    # candidate с другим числом запросов (2 vs 1)
    cand = _mock_result({"a": 1}, [["a"], ["b"]], cov=1.0, dm=0.0)
    try:
        diff_reports(base, cand)
        assert False
    except ValueError:
        pass


def test_diff_reports_coverage_drop():
    base = _mock_result({"a": 5, "b": 3, "c": 2, "d": 2}, [["a", "b"]], cov=1.0, dm=0.0)
    cand = _mock_result({"a": 5, "b": 3, "c": 0, "d": 0}, [["a", "b"]], cov=0.5, dm=0.5)
    d = diff_reports(base, cand)
    assert d.coverage_delta == -0.5
    assert d.dark_matter_delta == 0.5
    assert "c" in d.new_dark_matter
    assert "d" in d.new_dark_matter
    assert d.worst_losses[0][0] in ("c", "d")


def test_diff_reports_to_dict():
    base = _mock_result({"a": 5, "b": 3}, [["a", "b"]], cov=1.0, dm=0.0)
    cand = _mock_result({"a": 5, "b": 3}, [["a", "b"]], cov=1.0, dm=0.0)
    d = diff_reports(base, cand)
    dd = d.to_dict()
    assert dd["coverage_delta"] == 0.0
    assert dd["new_dark_matter"] == []


if __name__ == "__main__":
    import sys

    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    p = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            p += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
    print(f"\n{p}/{len(fns)} passed")
    sys.exit(0 if p == len(fns) else 1)
