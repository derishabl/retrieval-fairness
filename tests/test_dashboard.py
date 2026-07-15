"""test_dashboard.py — HTML dashboard generation."""

from __future__ import annotations

import os
import tempfile

from retrieval_fairness.dashboard import build_html, render_dashboard
from retrieval_fairness.probe import probe
from retrieval_fairness.stores import InMemoryVectorStore
from retrieval_fairness.types import Chunk, Query


def _toy():
    chunks = [
        Chunk(id="A", text="a", vector=[1.0, 0.0]),
        Chunk(id="B", text="b", vector=[1.0, 0.01]),
        Chunk(id="C", text="c", vector=[0.0, 1.0]),
        Chunk(id="D", text="d", vector=[0.0, -1.0]),
    ]
    queries = [Query(id="q1", vector=[1.0, 0.0]), Query(id="q2", vector=[1.0, 0.005])]
    return chunks, queries


def test_dashboard_writes_html_file():
    chunks, queries = _toy()
    store = InMemoryVectorStore(chunks)
    result = probe(store, queries, top_k=2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        render_dashboard(
            result, path, chunks_vectors=[c.vector for c in chunks], chunk_ids=[c.id for c in chunks]
        )
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as file:
            content = file.read()
    assert "<svg" in content  # есть Lorenz/histogram
    assert "Coverage" in content
    assert "A" in content  # хаб в таблице


def test_build_html_without_vectors():
    chunks, queries = _toy()
    result = probe(InMemoryVectorStore(chunks), queries, top_k=2)
    html_str = build_html(result)  # без PCA
    assert "<svg" in html_str
    assert "PCA" not in html_str  # проекции нет без векторов


def test_build_html_with_pca():
    chunks, queries = _toy()
    result = probe(InMemoryVectorStore(chunks), queries, top_k=2)
    html_str = build_html(result, chunks_vectors=[c.vector for c in chunks], chunk_ids=[c.id for c in chunks])
    assert "PCA" in html_str


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
