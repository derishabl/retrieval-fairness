"""
test_probe.py — тест прогона probe на in-memory сторе.

Маленький синтетический корпус + запросы; проверяем, что probe
корректно собирает частоты и отчёт, включая dark matter.
"""

from __future__ import annotations

from retrieval_fairness.probe import probe
from retrieval_fairness.stores import InMemoryVectorStore
from retrieval_fairness.types import Chunk, Query


def _toy_corpus() -> list[Chunk]:
    # 3 чанка на прямой оси: A в 0, B в 1, C далеко в 100
    return [
        Chunk(id="A", text="a", vector=[1.0, 0.0]),
        Chunk(id="B", text="b", vector=[1.0, 0.01]),
        Chunk(id="C", text="c", vector=[0.0, 1.0]),
        Chunk(id="D", text="d", vector=[0.0, -1.0]),  # далеко от всех запросов
    ]


def _toy_queries() -> list[Query]:
    return [
        Query(id="q1", vector=[1.0, 0.0]),
        Query(id="q2", vector=[1.0, 0.005]),
    ]


def test_probe_collects_freqs():
    store = InMemoryVectorStore(_toy_corpus())
    res = probe(store, _toy_queries(), top_k=2)
    # A и B всегда рядом с q1/q2 -> A и B найдены; C, D — dark matter
    assert res.freqs["A"] == 2
    assert res.freqs["B"] == 2
    assert res.freqs["C"] == 0
    assert res.freqs["D"] == 0
    assert res.report is not None
    assert res.report.coverage_pct == 0.5
    assert "D" in res.report.dark_matter_ids


def test_probe_respects_top_k():
    store = InMemoryVectorStore(_toy_corpus())
    res = probe(store, _toy_queries(), top_k=1)
    # при top_k=1 каждый запрос даёт 1 hit; всего 2 hit
    assert sum(res.freqs.values()) == 2


def test_probe_explicit_corpus_ids():
    store = InMemoryVectorStore(_toy_corpus())
    # передаём corpus_ids явно, включая id, которого нет в сторе
    res = probe(store, _toy_queries(), top_k=2, corpus_ids=["A", "B", "C", "D", "GHOST"])
    assert "GHOST" in res.freqs
    assert res.freqs["GHOST"] == 0
    assert res.report.n_chunks == 5


if __name__ == "__main__":
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
