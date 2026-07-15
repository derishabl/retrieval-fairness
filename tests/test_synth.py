"""test_synth.py — synthetic query generation."""

from __future__ import annotations

from retrieval_fairness.synth import synth_probe, synth_queries_from_corpus
from retrieval_fairness.types import Chunk


def _chunks():
    return [
        Chunk(id="a", text="Отпуск оформляется через HR-портал за две недели", vector=[1.0, 0.0]),
        Chunk(id="b", text="VPN настраивается через корпоративное приложение", vector=[0.0, 1.0]),
        Chunk(id="c", text="Зарплата выплачивается двумя частями каждый месяц", vector=[0.5, 0.5]),
    ]


def test_synth_generates_one_query_per_chunk():
    chunks = _chunks()
    queries, _vec = synth_queries_from_corpus(chunks, n_per_chunk=1)
    assert len(queries) == 3
    assert all(q.vector for q in queries)
    # id содержит id чанка
    assert queries[0].id.startswith("synth_a_")


def test_synth_n_per_chunk():
    chunks = _chunks()
    queries, _ = synth_queries_from_corpus(chunks, n_per_chunk=2)
    assert len(queries) == 6


def test_synth_keywords_nonempty():
    chunks = _chunks()
    queries, _ = synth_queries_from_corpus(chunks, query_style="keywords", n_terms=3)
    assert all(q.text.strip() for q in queries)


def test_synth_text_style():
    chunks = _chunks()
    queries, _ = synth_queries_from_corpus(chunks, query_style="text")
    # text style = первые 60 символов
    assert queries[0].text == chunks[0].text[:60]


def test_synth_probe_returns_report():
    chunks = _chunks()
    result = synth_probe(chunks, top_k=2, n_per_chunk=1)
    assert result.report is not None
    assert result.report.n_queries == 3
    # при синтетике «целевой запрос на чанк» coverage должна быть высокой
    assert result.report.coverage_pct >= 2 / 3


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
