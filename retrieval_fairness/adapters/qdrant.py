"""
adapters/qdrant.py — Qdrant-адаптер.

Работает поверх Qdrant (local или cloud) через qdrant-client.
Зависимость: pip install 'retrieval-fairness[qdrant]'

Тесты skip'аются без QDRANT_TEST_URL.
"""

from __future__ import annotations

from collections.abc import Iterator

from retrieval_fairness.adapters.base import BaseVectorStoreAdapter
from retrieval_fairness.types import Hit


class QdrantAdapter(BaseVectorStoreAdapter):
    """
    Qdrant-адаптер.

    url: Qdrant endpoint (http://localhost:6333).
    collection: имя коллекции.
    api_key: для Qdrant Cloud (опционально).
    vector_name: имя вектора в коллекции (для named-vectors; None = default).
    """

    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        vector_name: str | None = None,
    ):
        super().__init__()
        try:
            from qdrant_client import QdrantClient  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "QdrantAdapter requires qdrant-client: pip install 'retrieval-fairness[qdrant]'"
            ) from e
        self._url = url
        self._collection = collection
        self._api_key = api_key
        self._vector_name = vector_name
        self._client_instance = None  # ленивый персистентный клиент

    def _client(self):
        # Один клиент на весь workload: клиент-на-запрос доминировал бы
        # в накладных расходах на тысячах запросов.
        if self._client_instance is None:
            from qdrant_client import QdrantClient

            self._client_instance = QdrantClient(url=self._url, api_key=self._api_key)
        return self._client_instance

    def close(self) -> None:
        """Закрыть клиент (опционально; безопасно вызывать повторно)."""
        if self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None

    def _search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        from qdrant_client.models import SearchParams

        client = self._client()
        # client.search() deprecated в qdrant-client >=1.10 (убран в 2.x).
        # query_points() — современный API: query — сырой вектор,
        # named vector передаётся через using=. Совместимость: пробуем
        # новый, при AttributeError падаем на старый search().
        try:
            res = client.query_points(
                collection_name=self._collection,
                query=query_vec,
                using=self._vector_name,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128, exact=False),
            ).points
        except AttributeError:
            # старый qdrant-client (<1.10): нет query_points -> search()
            res = client.search(
                collection_name=self._collection,
                query_vector=query_vec if self._vector_name is None else (self._vector_name, query_vec),
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128, exact=False),
            )
        # Stabilize ordering among returned ties. Qdrant controls which IDs are
        # included at an approximate top-k boundary, so that limitation is also
        # recorded in provenance.
        selected = sorted(res, key=lambda point: (-float(point.score), str(point.id)))
        return [
            Hit(chunk_id=str(point.id), score=float(point.score), rank=rank)
            for rank, point in enumerate(selected, start=1)
        ]

    def _list_chunk_ids(self) -> Iterator[str]:
        # scroll API — пагинация по всему корпусу
        client = self._client()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self._collection,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for p in points:
                yield str(p.id)
            if offset is None:
                break

    def provenance_metadata(self) -> dict[str, object]:
        try:
            from importlib.metadata import version

            adapter_version = version("qdrant-client")
        except Exception:
            adapter_version = None
        return {
            "adapter": "qdrant",
            "adapter_version": adapter_version,
            "adapter_config": {
                "collection": self._collection,
                "vector_name": self._vector_name,
            },
            "distance_metric": None,
            "normalized": None,
            "search_params": {
                "hnsw_ef": 128,
                "exact": False,
                "tie_policy": "score_desc_chunk_id_asc",
                "boundary_ties_backend_limited": True,
            },
        }

    @property
    def size(self) -> int:
        info = self._client().get_collection(self._collection)
        return info.points_count or 0
