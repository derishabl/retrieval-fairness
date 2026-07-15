"""
types.py — контракты retrieval_fairness.

VectorStore — нейтральный интерфейс векторного хранилища. Любой стор
(FAISS, Qdrant, pgvector, ...) приводится к нему адаптером.
Раннеру всё равно, какой стор; метрики работают поверх search().
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Chunk:
    """Документ/чанк в корпусе."""

    id: str
    text: str
    vector: list[float] | None = None  # опционально — для in-memory


@dataclass(frozen=True)
class Hit:
    """Один результат поиска."""

    chunk_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class Query:
    """Запрос workload'а. Вектор — уже эмбеддированный."""

    id: str
    vector: list[float]
    text: str = ""


@runtime_checkable
class VectorStore(Protocol):
    """Минимальный контракт векторного хранилища."""

    def search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        """Вернуть top-k ближайших чанков к query_vec."""
        ...

    def list_chunks(self) -> Iterator[Chunk]:
        """Все чанки корпуса — для dark-matter/coverage."""
        ...
