"""
adapters/faiss.py — FAISS-адаптер.

Работает поверх faiss-cpu/faiss-gpu индекса + JSON sidecar с id-mapping
(FAISS хранит только векторы; id хранятся отдельно).

Полно тестируется локально (без сервера). В проде — индекс на диске.

Формат ids-map JSON: {"ids": ["a", "b", ...]} — по порядку строк индекса.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from retrieval_fairness.types import Hit
from retrieval_fairness.adapters.base import BaseVectorStoreAdapter
from retrieval_fairness.validation import validate_unique_ids, validate_vector


class FaissAdapter(BaseVectorStoreAdapter):
    """
    FAISS-адаптер.

    index_path: путь к .faiss индексу (IndexFlatIP/IndexFlatL2/...).
    ids_map_path: JSON {"ids": [...]} — id по порядку строк индекса.
    """

    def __init__(self, index_path: str, ids_map_path: str | None = None):
        super().__init__()
        import faiss

        self._faiss = faiss
        self._index = faiss.read_index(index_path)
        if ids_map_path:
            with open(ids_map_path, encoding="utf-8") as f:
                self._ids: list[str] = json.load(f)["ids"]
        else:
            self._ids = [str(i) for i in range(self._index.ntotal)]
        if len(self._ids) != self._index.ntotal:
            raise ValueError(f"ids-map length {len(self._ids)} != index ntotal {self._index.ntotal}")
        if any(not isinstance(chunk_id, str) for chunk_id in self._ids):
            raise ValueError("FAISS ids-map must contain string IDs")
        validate_unique_ids(self._ids, name="FAISS IDs")

    def _search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        q = np.array([query_vec], dtype="float32")
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []
        scores, indices = self._index.search(q, k)
        out = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            out.append(Hit(chunk_id=self._ids[idx], score=float(score), rank=rank))
        return out

    def _list_chunk_ids(self) -> Iterator[str]:
        yield from self._ids

    @property
    def size(self) -> int:
        return self._index.ntotal


def build_flat_index(
    vectors: list[list[float]], ids: list[str], index_path: str, ids_map_path: str, metric: str = "ip"
) -> None:
    """
    Утилита: построить IndexFlatIP/L2 + ids-map из векторов.
    metric: 'ip' (inner product / cosine на нормализованных) | 'l2'.
    """
    if not vectors:
        raise ValueError("vectors must not be empty")
    if len(vectors) != len(ids):
        raise ValueError(f"len(vectors)={len(vectors)} != len(ids)={len(ids)}")
    validate_unique_ids(ids, name="FAISS IDs")
    if metric not in {"ip", "l2"}:
        raise ValueError(f"unknown metric: {metric}")
    dim = len(vectors[0])
    for index, vector in enumerate(vectors):
        validate_vector(vector, name=f"vectors[{index}]", dim=dim)
    for output in (index_path, ids_map_path):
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    import faiss

    mat = np.array(vectors, dtype="float32")
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(mat)
    faiss.write_index(index, index_path)
    with open(ids_map_path, "w", encoding="utf-8") as f:
        json.dump({"ids": ids}, f, ensure_ascii=False)
