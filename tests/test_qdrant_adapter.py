"""test_qdrant_adapter.py — Qdrant adapter tests (skip without QDRANT_TEST_URL)."""

from __future__ import annotations

import os

import pytest

QDR_URL = os.environ.get("QDRANT_TEST_URL")
QDR_COLL = os.environ.get("QDRANT_TEST_COLLECTION", "test_collection")

pytestmark = pytest.mark.skipif(
    not QDR_URL,
    reason="set QDRANT_TEST_URL to run Qdrant adapter tests (needs Qdrant instance)",
)


def test_qdrant_search_and_corpus_ids():
    from retrieval_fairness.adapters.qdrant import QdrantAdapter

    adapter = QdrantAdapter(url=QDR_URL, collection=QDR_COLL)
    hits = adapter.search([1.0, 0.0, 0.0], top_k=3)
    assert len(hits) <= 3
    ids = adapter.corpus_ids()
    assert isinstance(ids, list)


if __name__ == "__main__":
    import sys

    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    p = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            p += 1
        except (AssertionError, Exception) as e:
            print(f"  SKIP/FAIL  {name}: {e}")
    print(f"\n{p}/{len(fns)} passed")
    sys.exit(0)
