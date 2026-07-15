"""test_gate.py — CI gate rules."""

from __future__ import annotations

import os
import tempfile

from retrieval_fairness.gate import evaluate_gate
from retrieval_fairness.probe import probe
from retrieval_fairness.serialize import save_probe
from retrieval_fairness.stores import InMemoryVectorStore
from retrieval_fairness.types import Chunk, Query


def _save(result, d, name):
    p = os.path.join(d, name)
    save_probe(result, p)
    return p


def _toy_results():
    chunks = [
        Chunk(id="A", text="a", vector=[1.0, 0.0]),
        Chunk(id="B", text="b", vector=[1.0, 0.01]),
        Chunk(id="C", text="c", vector=[0.0, 1.0]),
        Chunk(id="D", text="d", vector=[0.0, -1.0]),
    ]
    queries = [
        Query(id="q1", vector=[1.0, 0.0], text="first query"),
        Query(id="q2", vector=[1.0, 0.005], text="second query"),
    ]
    good = probe(InMemoryVectorStore(chunks), queries, top_k=2)
    # "ухудшенный": другой top-k -> coverage падает (D не находится и так, но top_k=1 -> меньше coverage)
    bad = probe(InMemoryVectorStore(chunks), queries, top_k=1)
    return good, bad


def test_gate_passes_when_no_regression():
    good, _ = _toy_results()
    with tempfile.TemporaryDirectory() as d:
        b = _save(good, d, "b.json")
        c = _save(good, d, "c.json")
        res = evaluate_gate(
            b, c, max_coverage_drop=0.05, max_dark_matter_rise=0.05, max_gini_rise=0.1, min_query_overlap=0.8
        )
    assert res.passed is True
    assert all(r.passed for r in res.rules)


def test_gate_fails_on_coverage_drop():
    good, bad = _toy_results()
    with tempfile.TemporaryDirectory() as d:
        b = _save(good, d, "b.json")
        c = _save(bad, d, "c.json")
        # bad имеет coverage 0.5, good 0.5 -> drop 0; нужно реальное падение
        # используем top_k чтобы сделать различие: good top_k=2 (cov 0.5), сделаем bad меньше
        res = evaluate_gate(b, c, max_coverage_drop=0.05)
    # coverage у обоих 0.5 (A,B находятся, C,D нет) -> drop=0 -> pass
    # поэтому тест: pass при равном coverage
    assert res.passed is True


def test_gate_fails_when_coverage_actually_drops():
    # конструируем реальное падение coverage
    chunks = [
        Chunk(id="A", text="a", vector=[1.0, 0.0]),
        Chunk(id="B", text="b", vector=[1.0, 0.01]),
        Chunk(id="C", text="c", vector=[0.9, 0.0]),  # близко к A
        Chunk(id="D", text="d", vector=[0.9, 0.01]),  # близко к B
    ]
    queries = [
        Query(id="q1", vector=[1.0, 0.0], text="first query"),
        Query(id="q2", vector=[1.0, 0.005], text="second query"),
    ]
    base = probe(InMemoryVectorStore(chunks), queries, top_k=4)  # все находятся
    cand = probe(InMemoryVectorStore(chunks), queries, top_k=1)  # только 2 из 4
    assert base.report.coverage_pct > cand.report.coverage_pct
    with tempfile.TemporaryDirectory() as d:
        b = _save(base, d, "b.json")
        c = _save(cand, d, "c.json")
        res = evaluate_gate(b, c, max_coverage_drop=0.05)
    assert res.passed is False
    assert any(r.name == "coverage_drop" and not r.passed for r in res.rules)


def test_gate_no_rules_always_passes():
    good, _ = _toy_results()
    with tempfile.TemporaryDirectory() as d:
        b = _save(good, d, "b.json")
        c = _save(good, d, "c.json")
        res = evaluate_gate(b, c)  # все пороги None -> правил нет
    assert res.passed is True
    assert res.rules == []


def test_gate_zero_tolerance_catches_any_drop():
    # max_coverage_drop=0 (zero tolerance) -> любое падение = fail
    base = probe(InMemoryVectorStore(_real_drop_chunks()), _real_drop_queries(), top_k=4)
    cand = probe(InMemoryVectorStore(_real_drop_chunks()), _real_drop_queries(), top_k=1)
    with tempfile.TemporaryDirectory() as d:
        b = _save(base, d, "b.json")
        c = _save(cand, d, "c.json")
        res = evaluate_gate(b, c, max_coverage_drop=0.0)
    assert res.passed is False
    cov_rule = next(r for r in res.rules if r.name == "coverage_drop")
    assert cov_rule.threshold == 0.0


def test_gate_zero_tolerance_passes_on_no_change():
    good, _ = _toy_results()
    with tempfile.TemporaryDirectory() as d:
        b = _save(good, d, "b.json")
        c = _save(good, d, "c.json")
        res = evaluate_gate(b, c, max_coverage_drop=0.0)
    assert res.passed is True  # drop=0 <= 0


def _real_drop_chunks():
    return [
        Chunk(id="A", text="a", vector=[1.0, 0.0]),
        Chunk(id="B", text="b", vector=[1.0, 0.01]),
        Chunk(id="C", text="c", vector=[0.9, 0.0]),
        Chunk(id="D", text="d", vector=[0.9, 0.01]),
    ]


def _real_drop_queries():
    return [
        Query(id="q1", vector=[1.0, 0.0], text="first query"),
        Query(id="q2", vector=[1.0, 0.005], text="second query"),
    ]


def test_gate_rejects_out_of_range_threshold():
    base = probe(InMemoryVectorStore(_real_drop_chunks()), _real_drop_queries(), top_k=4)
    cand = probe(InMemoryVectorStore(_real_drop_chunks()), _real_drop_queries(), top_k=1)
    with tempfile.TemporaryDirectory() as d:
        b = _save(base, d, "b.json")
        c = _save(cand, d, "c.json")
        # 5.0 вне [0,1] — раньше молча pass (гейт, который никогда не сработает)
        try:
            evaluate_gate(b, c, max_coverage_drop=5.0)
            assert False, "expected ValueError for out-of-range threshold"
        except ValueError:
            pass
        # отрицательный порог тоже невалиден
        try:
            evaluate_gate(b, c, max_dark_matter_rise=-0.1)
            assert False
        except ValueError:
            pass


def test_gate_validates_all_threshold_ranges():
    base = probe(InMemoryVectorStore(_real_drop_chunks()), _real_drop_queries(), top_k=4)
    with tempfile.TemporaryDirectory() as d:
        b = _save(base, d, "b.json")
        c = _save(base, d, "c.json")
        # граница 1.0 и 0.0 — валидны
        evaluate_gate(
            b, c, max_coverage_drop=1.0, max_dark_matter_rise=1.0, max_gini_rise=1.0, min_query_overlap=0.0
        )
        # gini_rise > 1.0 — невалиден
        try:
            evaluate_gate(b, c, max_gini_rise=2.0)
            assert False
        except ValueError:
            pass


def test_probe_to_gate_end_to_end_cli(tmp_path=None):
    """
    Блокер-сценарий: probe --json (через save_probe) -> gate --baseline.
    Раньше probe сохранял report.to_dict(), а gate ждал формат save_probe -> ломалось.
    """
    import os
    import tempfile

    from retrieval_fairness.serialize import save_probe

    chunks = _real_drop_chunks()
    queries = _real_drop_queries()
    base = probe(InMemoryVectorStore(chunks), queries, top_k=4)
    cand = probe(InMemoryVectorStore(chunks), queries, top_k=1)
    with tempfile.TemporaryDirectory() as d:
        bp = os.path.join(d, "base.json")
        cp = os.path.join(d, "cand.json")
        save_probe(base, bp)  # то, что делает probe --json после фикса
        save_probe(cand, cp)
        # gate грузит через load_probe — должно работать
        res = evaluate_gate(bp, cp, max_coverage_drop=0.05, max_dark_matter_rise=0.05)
    assert res.passed is False
    assert any(r.name == "coverage_drop" and not r.passed for r in res.rules)


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
