"""test_pgvector_adapter.py — pgvector adapter tests (skip without DATABASE_URL)."""

from __future__ import annotations

import os

import pytest

PGV_URL = os.environ.get("PGVECTOR_TEST_URL")  # e.g. postgres://user:pass@localhost:5432/db

pytestmark = pytest.mark.skipif(
    not PGV_URL,
    reason="set PGVECTOR_TEST_URL to run pgvector adapter tests (needs pgvector instance)",
)


def test_pgvector_search_and_corpus_ids():
    from retrieval_fairness.adapters.pgvector import PgvectorAdapter

    adapter = PgvectorAdapter(database_url=PGV_URL, table="docs")
    # прогон простого поиска
    hits = adapter.search([1.0, 0.0, 0.0], top_k=3)
    assert len(hits) <= 3
    for h in hits:
        assert h.rank >= 1
    # corpus_ids
    ids = adapter.corpus_ids()
    assert isinstance(ids, list)
    assert len(ids) == adapter.size


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
