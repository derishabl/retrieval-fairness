"""test_faiss_adapter.py — FAISS adapter vs InMemory consistency."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from retrieval_fairness.adapters.faiss import FaissAdapter, build_flat_index
from retrieval_fairness.adapters.inmemory import InMemoryVectorStore
from retrieval_fairness.probe import probe
from retrieval_fairness.types import Chunk, Query

pytest.importorskip("faiss", reason="FAISS adapter extra is not installed")


def _toy():
    # нормализованные векторы, чтобы IP == cosine
    vecs = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.01],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.9, 0.1],
        ],
        dtype="float32",
    )
    ids = ["a", "b", "c", "d", "e"]
    queries = [
        Query(id="q1", vector=[1.0, 0.0]),
        Query(id="q2", vector=[1.0, 0.005]),
        Query(id="q3", vector=[0.0, 1.0]),
    ]
    return vecs, ids, queries


def test_faiss_search_basic():
    vecs, ids, _ = _toy()
    with tempfile.TemporaryDirectory() as d:
        ip = os.path.join(d, "idx.faiss")
        mp = os.path.join(d, "ids.json")
        build_flat_index(vecs.tolist(), ids, ip, mp, metric="ip")
        adapter = FaissAdapter(ip, mp)
    hits = adapter.search([1.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id in ("a", "b")  # ближайшие к [1,0]
    assert hits[0].rank == 1


def test_faiss_corpus_ids():
    vecs, ids, _ = _toy()
    with tempfile.TemporaryDirectory() as d:
        ip = os.path.join(d, "idx.faiss")
        mp = os.path.join(d, "ids.json")
        build_flat_index(vecs.tolist(), ids, ip, mp)
        adapter = FaissAdapter(ip, mp)
    assert adapter.corpus_ids() == ["a", "b", "c", "d", "e"]


def test_faiss_metrics_match_inmemory():
    """Ключевой тест: probe на FAISS даёт те же метрики, что InMemory на тех же векторах."""
    vecs, ids, queries = _toy()
    chunks = [Chunk(id=i, text="t", vector=v.tolist()) for i, v in zip(ids, vecs)]
    inmem = probe(InMemoryVectorStore(chunks), queries, top_k=2)

    with tempfile.TemporaryDirectory() as d:
        ip = os.path.join(d, "idx.faiss")
        mp = os.path.join(d, "ids.json")
        build_flat_index(vecs.tolist(), ids, ip, mp, metric="ip")
        adapter = FaissAdapter(ip, mp)
    faiss_result = probe(adapter, queries, top_k=2)

    # метрики должны совпадать (IP на нормализованных == cosine)
    assert faiss_result.report.coverage_pct == inmem.report.coverage_pct
    assert faiss_result.report.dark_matter_pct == inmem.report.dark_matter_pct
    assert abs(faiss_result.report.gini - inmem.report.gini) < 1e-6
    # частоты по чанкам совпадают
    assert faiss_result.freqs == inmem.freqs


def test_faiss_manifest_rejects_other_index_of_same_length():
    vecs, ids, _ = _toy()
    with tempfile.TemporaryDirectory() as d:
        first_index = os.path.join(d, "first.faiss")
        first_manifest = os.path.join(d, "first.json")
        second_index = os.path.join(d, "second.faiss")
        second_manifest = os.path.join(d, "second.json")
        build_flat_index(vecs.tolist(), ids, first_index, first_manifest)
        build_flat_index((vecs * 0.5).tolist(), ids, second_index, second_manifest)
        with pytest.raises(ValueError, match="index_sha256"):
            FaissAdapter(first_index, second_manifest)


def test_faiss_validates_query_dimension_before_native_search():
    vecs, ids, _ = _toy()
    with tempfile.TemporaryDirectory() as d:
        index_path = os.path.join(d, "idx.faiss")
        manifest_path = os.path.join(d, "ids.json")
        build_flat_index(vecs.tolist(), ids, index_path, manifest_path)
        adapter = FaissAdapter(index_path, manifest_path)
        with pytest.raises(ValueError, match="dimension"):
            adapter.search([1.0, 2.0, 3.0], top_k=2)


def test_faiss_ids_map_length_mismatch():
    vecs, ids, _ = _toy()
    with tempfile.TemporaryDirectory() as d:
        ip = os.path.join(d, "idx.faiss")
        mp = os.path.join(d, "ids.json")
        build_flat_index(vecs.tolist(), ids, ip, mp)
        # испортить ids-map
        import json

        with open(mp, "w") as f:
            json.dump({"ids": ["a", "b"]}, f)  # длина 2, индекс 5
        try:
            FaissAdapter(ip, mp)
            assert False, "expected ValueError"
        except ValueError:
            pass


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
