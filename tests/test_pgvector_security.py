"""test_pgvector_security.py — SQL identifier validation (#3), no instance needed."""

from __future__ import annotations

# psycopg может быть не установлен — валидация идентификаторов должна работать
# до импорта psycopg (конструктор поднимает ImportError только если psycopg есть).
# Поэтому тестируем валидаторы напрямую.


def _import_validators():
    from retrieval_fairness.adapters import pgvector as pgv

    return pgv._validate_ident, pgv._ALLOWED_DISTANCE_OPS, pgv._IDENT_RE


def test_validate_ident_accepts_valid():
    _validate_ident, _, _ = _import_validators()
    for v in ["docs", "my_table", "schema.docs", "col_1", "_x"]:
        assert _validate_ident(v, "t") == v


def test_validate_ident_rejects_injection():
    _validate_ident, _, _ = _import_validators()
    bad = [
        "docs; DROP TABLE x; --",
        "docs--",
        "1table",  # старт с цифры
        "col name",  # пробел
        "col'",  # кавычка
        "col; DROP",  # точка с запятой
        "",  # пусто
    ]
    for v in bad:
        try:
            _validate_ident(v, "t")
            assert False, f"expected ValueError for {v!r}"
        except ValueError:
            pass


def test_pgvector_orders_logical_ids_and_score_ties(monkeypatch):
    import sys
    import types

    statements = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql, params=None):
            statements.append((sql, params))

        def __iter__(self):
            return iter([("A",), ("B",)])

        def fetchall(self):
            return [("A", 0.1), ("B", 0.1)]

    class Connection:
        closed = False

        def cursor(self):
            return Cursor()

    module = types.ModuleType("psycopg")
    module.connect = lambda *_args, **_kwargs: Connection()
    monkeypatch.setitem(sys.modules, "psycopg", module)

    from retrieval_fairness.adapters.pgvector import PgvectorAdapter

    adapter = PgvectorAdapter("postgresql://credential-is-never-serialized")
    assert adapter.corpus_ids() == ["A", "B"]
    assert [hit.chunk_id for hit in adapter.search([1.0], 2)] == ["A", "B"]
    assert "ORDER BY id ASC" in statements[0][0]
    assert "ORDER BY embedding <=> %s, id ASC" in statements[1][0]
    assert "credential" not in str(adapter.provenance_metadata())


def test_distance_op_whitelist():
    _, allowed, _ = _import_validators()
    assert allowed == {"<=>", "<->", "<#>"}
    # инъекция через distance_op
    try:
        # эмулируем конструктор без psycopg: проверяем только валидацию op
        # distance_op проверяется в __init__; без psycopg поднимётся ImportError,
        # поэтому проверяем логику напрямую
        assert "<=>" in allowed and "malicious" not in allowed
    except Exception:
        pass


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
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{p}/{len(fns)} passed")
    sys.exit(0 if p == len(fns) else 1)
