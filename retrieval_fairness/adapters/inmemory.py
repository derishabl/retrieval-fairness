"""
adapters/inmemory.py — in-memory векторный стор (косинус, numpy).

Лёгкий стор для разработки/тестов/демо: работает без внешних БД.
В проде заменяется FAISS/Qdrant/pgvector адаптерами по тому же контракту.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from retrieval_fairness.adapters.base import BaseVectorStoreAdapter
from retrieval_fairness.types import Chunk, Hit
from retrieval_fairness.validation import validate_unique_ids, validate_vector


def _cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.array([])
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-12:
        q_norm = 1e-12
    m_norm = np.linalg.norm(matrix, axis=1)
    m_norm = np.where(m_norm < 1e-12, 1e-12, m_norm)
    return (matrix @ query) / (m_norm * q_norm + 1e-12)


class InMemoryVectorStore(BaseVectorStoreAdapter):
    """
    Векторный стор в памяти: косинусный поиск по numpy-матрице.
    chunks: list[Chunk] — корпус; Chunk.vector обязан быть заполнен.
    """

    def __init__(self, chunks: list[Chunk]):
        super().__init__()
        self._chunks = list(chunks)
        self._ids = [c.id for c in self._chunks]
        validate_unique_ids(self._ids, name="corpus IDs")
        dimensions: int | None = None
        for index, chunk in enumerate(self._chunks):
            if chunk.vector is None:
                raise ValueError(f"chunks[{index}].vector is required")
            validate_vector(chunk.vector, name=f"chunks[{index}].vector", dim=dimensions)
            dimensions = len(chunk.vector) if dimensions is None else dimensions
        self._dim = dimensions
        self._matrix = (
            np.array([c.vector for c in self._chunks], dtype=float)
            if self._chunks
            else np.empty((0, 0), dtype=float)
        )

    def _search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        if self._dim is not None:
            validate_vector(query_vec, name="query vector", dim=self._dim)
        sims = _cosine_matrix(np.array(query_vec, dtype=float), self._matrix)
        if sims.size == 0:
            return []
        k = min(top_k, sims.size)
        # Public tie policy: score DESC, chunk_id ASC. A full lexicographic
        # order also makes the top-k boundary deterministic for duplicate and
        # zero vectors across NumPy versions/processes.
        idx = sorted(range(sims.size), key=lambda position: (-float(sims[position]), self._ids[position]))[:k]
        return [Hit(chunk_id=self._ids[i], score=float(sims[i]), rank=rank + 1) for rank, i in enumerate(idx)]

    def _list_chunk_ids(self) -> Iterator[str]:
        yield from self._ids

    def list_chunks(self) -> Iterator[Chunk]:
        yield from self._chunks

    def provenance_metadata(self) -> dict[str, object]:
        return {
            "adapter": "inmemory",
            "adapter_version": np.__version__,
            "adapter_config": {"corpus_size": len(self._chunks), "dimension": self._dim},
            "distance_metric": "cosine",
            "normalized": True,
            "search_params": {"tie_policy": "score_desc_chunk_id_asc"},
        }

    @property
    def size(self) -> int:
        return len(self._chunks)
