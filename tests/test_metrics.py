"""
test_metrics.py — юнит-тесты метрик exposure.

Граничные случаи + базовая корректность Gini/coverage/dark_matter/hub_capture.
"""

from __future__ import annotations

from retrieval_fairness.metrics import (
    build_report,
    coverage,
    dark_matter,
    gini,
    hub_capture,
    lorenz,
    retrieval_frequencies,
)


def test_coverage_full():
    freqs = {"a": 3, "b": 1, "c": 2}
    assert coverage(freqs) == 1.0
    assert dark_matter(freqs) == 0.0


def test_coverage_partial():
    freqs = {"a": 3, "b": 0, "c": 2, "d": 0}
    assert coverage(freqs) == 0.5
    assert dark_matter(freqs) == 0.5


def test_coverage_empty():
    assert coverage({}) == 0.0
    assert dark_matter({}) == 0.0


def test_gini_uniform():
    # все равны -> Gini = 0
    freqs = {f"c{i}": 5 for i in range(10)}
    assert abs(gini(freqs) - 0.0) < 1e-9


def test_gini_maximal():
    # всё в одном чанке -> Gini -> 1 - 1/n (максимум для распределения)
    n = 100
    freqs = {f"c{i}": (1000 if i == 0 else 0) for i in range(n)}
    g = gini(freqs)
    assert g > 0.98  # близко к 1


def test_gini_empty_and_zero():
    assert gini({}) == 0.0
    assert gini({"a": 0, "b": 0}) == 0.0  # ничего не находится


def test_hub_capture_all_in_one():
    freqs = {"a": 100, "b": 0, "c": 0, "d": 0, "e": 0}
    assert hub_capture(freqs, top_n=5) == 1.0
    assert hub_capture(freqs, top_n=1) == 1.0


def test_hub_capture_uniform():
    freqs = {f"c{i}": 10 for i in range(20)}
    # top-5 из 20 равных -> 5/20 = 0.25
    assert abs(hub_capture(freqs, top_n=5) - 0.25) < 1e-9


def test_lorenz_endpoints():
    freqs = {"a": 3, "b": 1, "c": 2}
    pts = lorenz(freqs)
    assert pts[0] == (0.0, 0.0)
    assert abs(pts[-1][1] - 1.0) < 1e-9
    # x монотонно не убывает
    xs = [p[0] for p in pts]
    assert all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))


def test_retrieval_frequencies_includes_dark_matter():
    corpus = ["a", "b", "c", "d"]
    hits_per_query = [["a", "b"], ["a", "c"]]
    freqs = retrieval_frequencies(hits_per_query, corpus)
    assert freqs == {"a": 2, "b": 1, "c": 1, "d": 0}


def test_build_report_fields():
    freqs = {"a": 5, "b": 3, "c": 0, "d": 0}
    rep = build_report(freqs, n_queries=10, top_k=5)
    assert rep.n_chunks == 4
    assert rep.coverage_pct == 0.5
    assert rep.dark_matter_pct == 0.5
    assert len(rep.dark_matter_ids) == 2
    assert rep.hub_leaderboard[0][0] == "a"
    d = rep.to_dict()
    assert d["coverage_pct"] == 0.5
    assert d["dark_matter_count"] == 2


if __name__ == "__main__":
    # простой runner без pytest
    import sys

    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)


def test_reachability_ceiling_bounds():
    from retrieval_fairness.metrics import reachability_ceiling

    # нельзя найти больше уникальных чанков, чем n_queries*top_k, и не больше корпуса
    assert reachability_ceiling(260000, 3452, 10) == 34520
    # если корпус меньше потолка — потолок ограничен корпусом
    assert reachability_ceiling(31, 11, 5) == 31  # 31 < 55
    assert reachability_ceiling(100, 50, 10) == 100  # 100 < 500
    # потолок 0 — не найти ничего нельзя
    assert reachability_ceiling(1000, 0, 10) == 0


def test_coverage_of_ceiling_vs_coverage():
    """На 260k-подобной ситуации coverage ~11.7% = ~88% от workload-потолка."""
    from retrieval_fairness.metrics import build_report

    # 100 чанков, 5 запросов, top-4 -> потолок 20; найдено 10 -> coverage 10%, of ceiling 50%
    freqs = {f"c{i}": (1 if i < 10 else 0) for i in range(100)}
    rep = build_report(freqs, n_queries=5, top_k=4)
    assert rep.reachability_ceiling == 20
    assert abs(rep.coverage_pct - 0.10) < 1e-6
    assert abs(rep.coverage_of_ceiling - 0.50) < 1e-6
    # to_dict содержит оба новых поля
    d = rep.to_dict()
    assert d["reachability_ceiling"] == 20
    assert d["coverage_of_ceiling"] == 0.5
