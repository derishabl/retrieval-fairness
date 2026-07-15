"""test_adapters.py — adapter registry and base infrastructure."""

from __future__ import annotations

from retrieval_fairness.adapters import BaseVectorStoreAdapter, InMemoryVectorStore, get_adapter
from retrieval_fairness.adapters.base import BaseVectorStoreAdapter as Base
from retrieval_fairness.types import Chunk


def test_get_adapter_inmemory():
    chunks = [Chunk(id="a", text="x", vector=[1.0, 0.0])]
    adapter = get_adapter("inmemory", chunks=chunks)
    assert isinstance(adapter, InMemoryVectorStore)
    assert isinstance(adapter, BaseVectorStoreAdapter)


def test_get_adapter_unknown_raises():
    try:
        get_adapter("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_corpus_ids_cached():
    chunks = [Chunk(id=f"c{i}", text="t", vector=[float(i), 0.0]) for i in range(3)]
    adapter = InMemoryVectorStore(chunks)
    ids1 = adapter.corpus_ids()
    ids2 = adapter.corpus_ids()
    assert ids1 == ["c0", "c1", "c2"]
    assert ids1 is ids2  # кэш


def test_base_search_not_implemented():
    class Dummy(Base):
        pass

    try:
        Dummy().search([1.0], 1)
        assert False
    except NotImplementedError:
        pass


def test_inmemory_is_base():
    chunks = [Chunk(id="a", text="x", vector=[1.0, 0.0])]
    assert isinstance(InMemoryVectorStore(chunks), BaseVectorStoreAdapter)


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
