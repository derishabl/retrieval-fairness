"""test_serialize.py — probe save/load round-trip."""

from __future__ import annotations

import os
import tempfile

from retrieval_fairness.probe import probe
from retrieval_fairness.serialize import load_probe, save_probe
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
    return InMemoryVectorStore(chunks), queries


def test_roundtrip_preserves_freqs_and_hits():
    store, queries = _toy()
    result = probe(store, queries, top_k=2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "baseline.json")
        save_probe(result, path)
        loaded = load_probe(path)
    assert loaded.freqs == result.freqs
    assert loaded.hits_per_query == result.hits_per_query
    assert loaded.report.coverage_pct == result.report.coverage_pct
    assert loaded.report.n_chunks == result.report.n_chunks
    assert loaded.report.dark_matter_ids == result.report.dark_matter_ids


def test_roundtrip_preserves_lorenz_and_dark_matter_ids():
    """#TODO-публикация: save/load был асимметричен — lorenz_curve терялась."""
    store, queries = _toy()
    result = probe(store, queries, top_k=2)
    assert result.report.lorenz_curve, "в оригинале кривая есть"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "b.json")
        save_probe(result, path)
        loaded = load_probe(path)
    assert loaded.report.lorenz_curve, "после load кривая не должна теряться"
    for (x0, y0), (x1, y1) in zip(result.report.lorenz_curve, loaded.report.lorenz_curve):
        assert abs(x0 - x1) < 1e-5 and abs(y0 - y1) < 1e-5
    # dark_matter_ids теперь явно в JSON-отчёте
    d2 = result.report.to_dict()
    assert d2["dark_matter_ids"] == result.report.dark_matter_ids
    assert d2["dark_matter_count"] == len(result.report.dark_matter_ids)


def test_roundtrip_diff_matches():
    """diff(original, loaded) должен быть нулевым."""
    from retrieval_fairness.diff import diff_reports

    store, queries = _toy()
    result = probe(store, queries, top_k=2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "b.json")
        save_probe(result, path)
        loaded = load_probe(path)
    d = diff_reports(result, loaded)
    assert d.coverage_delta == 0.0
    assert d.gini_delta == 0.0
    assert d.mean_query_overlap == 1.0


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
