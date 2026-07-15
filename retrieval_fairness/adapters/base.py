"""
adapters/base.py — общий базовый класс для адаптеров векторных хранилищ.

Содержит общую логику: кэш corpus_ids, валидация, хуки.
Конкретные адаптеры реализуют _search() и _list_chunk_ids().
"""

from __future__ import annotations

from collections.abc import Iterator

from retrieval_fairness.types import Chunk, Hit
from retrieval_fairness.validation import require_positive_int, validate_vector


class BaseVectorStoreAdapter:
    """Базовый класс адаптера. Подклассы реализуют _search и _list_chunk_ids."""

    def __init__(self) -> None:
        self._corpus_ids_cache: list[str] | None = None

    # -- contract: VectorStore-compatible --------------------------------
    def search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        require_positive_int(top_k, "top_k")
        validate_vector(query_vec, name="query vector")
        return self._search(query_vec, top_k)

    def list_chunks(self) -> Iterator[Chunk]:
        """
        По умолчанию возвращает Chunk без векторов (text недоступен во многих
        БД из контекста адаптера). Подклассы могут переопределить.
        """
        for cid in self._list_chunk_ids():
            yield Chunk(id=cid, text="", vector=None)

    # -- public helpers --------------------------------------------------
    def corpus_ids(self) -> list[str]:
        """Все id корпуса (кэшируется). Для dark-matter/coverage."""
        if self._corpus_ids_cache is None:
            self._corpus_ids_cache = list(self._list_chunk_ids())
        return self._corpus_ids_cache

    def provenance_metadata(self) -> dict[str, object]:
        """Credential-free adapter facts embedded into probe artifacts."""
        return {"adapter": type(self).__name__}

    # -- to override -----------------------------------------------------
    def _search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        raise NotImplementedError

    def _list_chunk_ids(self) -> Iterator[str]:
        raise NotImplementedError
