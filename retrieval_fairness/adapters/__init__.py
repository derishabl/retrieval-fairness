"""
retrieval_fairness.adapters — адаптеры векторных хранилищ.

Каждый адаптер приводит конкретный стор (FAISS, Qdrant, pgvector, ...)
к контракту VectorStore. Раннеру всё равно, какой стор; метрики
работают поверх search()/list_chunks().

Реестр get_adapter(name, **conn) — для CLI.
"""

from __future__ import annotations

from retrieval_fairness.adapters.base import BaseVectorStoreAdapter
from retrieval_fairness.adapters.inmemory import InMemoryVectorStore


def get_adapter(name: str, **conn) -> BaseVectorStoreAdapter:
    """
    Создать адаптер по имени + connection-аргументам.

    name: 'inmemory' | 'faiss' | 'pgvector' | 'qdrant'
    conn: depends on adapter (e.g. index_path/ids_map, database_url, ...)
    """
    name = name.lower()
    if name == "inmemory":
        # conn: chunks=[Chunk,...] (in-process)
        chunks = conn.get("chunks")
        if chunks is None:
            raise ValueError("inmemory adapter requires chunks=[Chunk,...]")
        return InMemoryVectorStore(chunks)
    if name == "faiss":
        from retrieval_fairness.adapters.faiss import FaissAdapter

        return FaissAdapter(index_path=conn["index_path"], ids_map_path=conn.get("ids_map_path"))
    if name == "pgvector":
        from retrieval_fairness.adapters.pgvector import PgvectorAdapter

        return PgvectorAdapter(
            database_url=conn["database_url"],
            table=conn.get("table", "docs"),
            column=conn.get("column", "embedding"),
        )
    if name == "qdrant":
        from retrieval_fairness.adapters.qdrant import QdrantAdapter

        return QdrantAdapter(url=conn["url"], collection=conn["collection"], api_key=conn.get("api_key"))
    raise ValueError(f"unknown adapter: {name}")


__all__ = ["BaseVectorStoreAdapter", "InMemoryVectorStore", "get_adapter"]
