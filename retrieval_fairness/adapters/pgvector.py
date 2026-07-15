"""
adapters/pgvector.py — pgvector-адаптер.

Работает поверх PostgreSQL + pgvector: поиск через оператор <=>
(cosine distance) или <-> (L2). Требует psycopg3.

Зависимость: pip install 'retrieval-fairness[pgvector]'

Тесты skip'аются без DATABASE_URL (нет инстанса). В CI — docker-compose
с pgvector (см. план §10.6).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from retrieval_fairness.adapters.base import BaseVectorStoreAdapter
from retrieval_fairness.types import Hit

# Идентификаторы SQL (table/column/id_column) вставляются в запрос через f-string,
# поэтому валидируем жёстко: только буквы/цифры/_/. и старт не с цифры.
# Это защита от SQL-инъекции через CLI --table/--column/--id-column.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_ALLOWED_DISTANCE_OPS = {"<=>", "<->", "<#>"}  # cosine / L2 / inner product


def _validate_ident(value: str, field: str) -> str:
    if not _IDENT_RE.fullmatch(value):
        raise ValueError(
            f"pgvector: {field}={value!r} не валидный SQL-идентификатор (допустимы [A-Za-z_][A-Za-z0-9_.]*)"
        )
    return value


class PgvectorAdapter(BaseVectorStoreAdapter):
    """
    pgvector-адаптер.

    database_url: psycopg connection string (postgres://...).
    table: таблица с документами (должна содержать id-колонку 'id' и
           векторную колонку column).
    column: имя векторной колонки.
    id_column: имя id-колонки (по умолчанию 'id').
    distance_op: '<=>' (cosine) | '<->' (L2) | '<#>' (inner product).
    """

    def __init__(
        self,
        database_url: str,
        table: str = "docs",
        column: str = "embedding",
        id_column: str = "id",
        distance_op: str = "<=>",
    ):
        super().__init__()
        try:
            import psycopg  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PgvectorAdapter requires psycopg: pip install 'retrieval-fairness[pgvector]'"
            ) from e
        self._database_url = database_url
        self._table = _validate_ident(table, "table")
        self._column = _validate_ident(column, "column")
        self._id_column = _validate_ident(id_column, "id_column")
        if distance_op not in _ALLOWED_DISTANCE_OPS:
            raise ValueError(
                f"pgvector: distance_op={distance_op!r} не разрешён "
                f"(допустимы {sorted(_ALLOWED_DISTANCE_OPS)})"
            )
        self._distance_op = distance_op
        self._conn = None  # ленивое персистентное соединение

    def _connect(self):
        # Одно соединение на весь workload: connect-на-запрос доминировал бы
        # в накладных расходах на тысячах запросов. Переподключаемся,
        # если соединение закрыто (таймаут/обрыв).
        import psycopg

        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._database_url, autocommit=True)
        return self._conn

    def close(self) -> None:
        """Закрыть соединение (опционально; безопасно вызывать повторно)."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def _search(self, query_vec: list[float], top_k: int) -> list[Hit]:
        # параметризуем вектор как строку '[v1,v2,...]' — pgvector парсит
        vec_str = "[" + ",".join(f"{v:.8g}" for v in query_vec) + "]"
        sql = (
            f"SELECT {self._id_column}, {self._column} {self._distance_op} %s AS dist "
            f"FROM {self._table} "
            f"ORDER BY {self._column} {self._distance_op} %s, {self._id_column} ASC LIMIT %s"
        )
        with self._connect().cursor() as cur:
            cur.execute(sql, (vec_str, vec_str, top_k))
            rows = cur.fetchall()
        out = []
        for rank, (cid, dist) in enumerate(rows, start=1):
            # для cosine dist (0=same, 2=opposite); score = 1 - dist/2 (~similarity)
            score = 1.0 - float(dist) / 2.0 if self._distance_op == "<=>" else -float(dist)
            out.append(Hit(chunk_id=str(cid), score=score, rank=rank))
        return out

    def _list_chunk_ids(self) -> Iterator[str]:
        with self._connect().cursor() as cur:
            cur.execute(f"SELECT {self._id_column} FROM {self._table} ORDER BY {self._id_column} ASC")
            for (cid,) in cur:  # итерация без fetchall — не держим все id в памяти дважды
                yield str(cid)

    def provenance_metadata(self) -> dict[str, object]:
        try:
            from importlib.metadata import version

            adapter_version = version("psycopg")
        except Exception:
            adapter_version = None
        metric = {"<=>": "cosine", "<->": "l2", "<#>": "inner_product"}[self._distance_op]
        return {
            "adapter": "pgvector",
            "adapter_version": adapter_version,
            "adapter_config": {
                "table": self._table,
                "column": self._column,
                "id_column": self._id_column,
            },
            "distance_metric": metric,
            "normalized": None,
            "search_params": {"tie_policy": "distance_asc_chunk_id_asc"},
        }

    @property
    def size(self) -> int:
        with self._connect().cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return cur.fetchone()[0]
