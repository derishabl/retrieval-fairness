"""
test_qdrant_mock.py — мок-тесты QdrantAdapter без сервера и без qdrant-client.

Реальный интеграционный тест скипается без QDRANT_TEST_URL, поэтому путь
формирования вызова query_points раньше не был покрыт вовсе (и содержал
несуществующий импорт Query + неверную передачу named vector). Здесь
подменяем qdrant_client фейковым модулем и проверяем контракт вызова:
  - query = сырой вектор (не обёртка)
  - named vector -> using=
  - клиент создаётся один раз на весь workload
"""

from __future__ import annotations

import sys
import types


def _install_fake_qdrant(monkeypatch, recorded: dict):
    qc_mod = types.ModuleType("qdrant_client")
    models_mod = types.ModuleType("qdrant_client.models")

    class SearchParams:
        def __init__(self, **kw):
            self.kw = kw

    class _Point:
        def __init__(self, id, score):
            self.id, self.score = id, score

    class _Resp:
        def __init__(self, points):
            self.points = points

    class QdrantClient:
        def __init__(self, url=None, api_key=None):
            recorded["n_clients"] = recorded.get("n_clients", 0) + 1
            recorded["url"] = url

        def query_points(self, collection_name, query, using, limit, search_params):
            recorded["call"] = {
                "collection_name": collection_name,
                "query": query,
                "using": using,
                "limit": limit,
            }
            return _Resp([_Point("a", 0.9), _Point("b", 0.5)])

        def close(self):
            recorded["closed"] = True

    qc_mod.QdrantClient = QdrantClient
    models_mod.SearchParams = SearchParams
    qc_mod.models = models_mod
    monkeypatch.setitem(sys.modules, "qdrant_client", qc_mod)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models_mod)


def test_search_passes_raw_vector_and_default_using(monkeypatch):
    recorded: dict = {}
    _install_fake_qdrant(monkeypatch, recorded)
    from retrieval_fairness.adapters.qdrant import QdrantAdapter

    a = QdrantAdapter(url="http://x:6333", collection="col")
    hits = a.search([0.1, 0.2], top_k=2)

    call = recorded["call"]
    assert call["query"] == [0.1, 0.2]  # сырой вектор, не обёртка
    assert call["using"] is None  # default vector
    assert call["collection_name"] == "col"
    assert call["limit"] == 2
    assert [h.chunk_id for h in hits] == ["a", "b"]
    assert [h.rank for h in hits] == [1, 2]


def test_search_named_vector_uses_using(monkeypatch):
    recorded: dict = {}
    _install_fake_qdrant(monkeypatch, recorded)
    from retrieval_fairness.adapters.qdrant import QdrantAdapter

    a = QdrantAdapter(url="http://x:6333", collection="col", vector_name="image")
    a.search([0.3], top_k=1)

    call = recorded["call"]
    assert call["query"] == [0.3]  # вектор НЕ заворачивается в dict
    assert call["using"] == "image"  # named vector -> using=


def test_provenance_excludes_endpoint_and_api_key(monkeypatch):
    recorded: dict = {}
    _install_fake_qdrant(monkeypatch, recorded)
    from retrieval_fairness.adapters.qdrant import QdrantAdapter

    adapter = QdrantAdapter(url="https://secret-host.example", collection="col", api_key="top-secret")
    metadata = str(adapter.provenance_metadata())
    assert "top-secret" not in metadata
    assert "secret-host" not in metadata
    assert "col" in metadata


def test_client_is_reused_across_searches(monkeypatch):
    recorded: dict = {}
    _install_fake_qdrant(monkeypatch, recorded)
    from retrieval_fairness.adapters.qdrant import QdrantAdapter

    a = QdrantAdapter(url="http://x:6333", collection="col")
    a.search([0.1], top_k=1)
    a.search([0.2], top_k=1)
    assert recorded["n_clients"] == 1  # один клиент на весь workload

    a.close()
    assert recorded.get("closed") is True
    a.close()  # повторный close безопасен
