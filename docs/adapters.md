# Adapters — подключи свой векторный стор

retrieval-fairness работает с любым векторным хранилищем через общий
контракт `VectorStore` (`search`, `list_chunks`). Выбор стор — флагом
`--store` в CLI или через `get_adapter(name, **conn)` в Python.

## Установка под конкретный стор

```bash
pip install -e .                    # ядро
pip install -e '.[faiss]'           # + FAISS
pip install -e '.[pgvector]'        # + pgvector
pip install -e '.[qdrant]'          # + Qdrant
pip install -e '.[models]'          # + sentence-transformers (для Шага 9)
```

## inmemory (по умолчанию, для дев/тестов)

```bash
retrieval-fairness probe --store inmemory --corpus corpus.jsonl \
    --queries queries.jsonl --top-k 10 --html dashboard.html
```

```python
from retrieval_fairness.adapters import InMemoryVectorStore
store = InMemoryVectorStore(chunks)   # list[Chunk] с vector
```

## FAISS

Нужен `.faiss` индекс + schema-v2 JSON manifest. Manifest хранит ordered
IDs, SHA-256 индекса, dimension, metric и normalization и поэтому отклоняет
sidecar от другого индекса даже при одинаковой длине. Построить пару:

```python
from retrieval_fairness.adapters.faiss import build_flat_index
build_flat_index(
    vectors, ids, "idx.faiss", "ids.json", metric="ip", normalized=True
)
# metric: 'ip' (inner product / cosine на нормализованных) | 'l2'
```

Прогон:

```bash
retrieval-fairness probe --store faiss --index-path idx.faiss \
    --ids-map ids.json --queries queries.jsonl --top-k 10
```

```python
from retrieval_fairness.adapters import get_adapter
store = get_adapter("faiss", index_path="idx.faiss", ids_map_path="ids.json")
```

## pgvector

PostgreSQL + расширение pgvector. Таблица должна иметь id-колонку и
векторную колонку типа `vector`/`halfvec`.

```bash
retrieval-fairness probe --store pgvector \
    --database-url "postgresql://user:pass@localhost:5432/db" \
    --table docs --column embedding --queries queries.jsonl --top-k 10
```

```python
store = get_adapter("pgvector", database_url="postgresql://...",
                    table="docs", column="embedding")
```

Оператор дистанции по умолчанию `<=>` (cosine); варианты `<->` (L2),
`<#>` (inner product) — через `distance_op=` в Python API. Logical corpus IDs
перечисляются с `ORDER BY id`; score ties разрешаются вторичным `id ASC`.

## Qdrant

```bash
retrieval-fairness probe --store qdrant \
    --url http://localhost:6333 --collection my_collection \
    --queries queries.jsonl --top-k 10
# для Qdrant Cloud: --api-key ...
```

```python
store = get_adapter("qdrant", url="http://localhost:6333",
                    collection="my_collection")
```

## Написать свой адаптер

Подкласс `BaseVectorStoreAdapter`, реализовать `_search` и
`_list_chunk_ids`:

```python
from retrieval_fairness.adapters.base import BaseVectorStoreAdapter
from retrieval_fairness.types import Hit

class MyAdapter(BaseVectorStoreAdapter):
    def __init__(self, ...):
        super().__init__()
        # подключение к вашему стору

    def _search(self, query_vec, top_k):
        # вернуть list[Hit] (chunk_id, score, rank)
        ...

    def _list_chunk_ids(self):
        # yield str id каждого чанка корпуса
        ...
```

После этого он работает с `probe`, `diff`, `gate`, `dashboard` без
дополнительной интеграции — контракт общий.

## Честные оговорки

- **Pinecone** — SaaS, нужен API key; адаптер в roadmap, не в v0.2.
- **Weaviate/Milvus/Chroma** — через PR; реестр `get_adapter` расширяется.
- **Тесты серверных адаптеров** (pgvector, Qdrant) — skip без инстанса
  (переменные `PGVECTOR_TEST_URL`, `QDRANT_TEST_URL`); в CI через
  docker-compose. FAISS тестируется полностью локально.
- **Score-шкалы** разных БД отличаются (cosine dist / L2 / IP) —
  метрики exposure работают по **рангам/top-k id**, не по score, поэтому
  сопоставимы между сторами.
- Публичная tie policy: score/distance, затем `chunk_id ASC`. InMemory и SQL
  детерминированы включая границу top-k. FAISS/Qdrant стабилизируют порядок
  возвращённых ties, но backend может сам выбрать boundary ties; версия backend
  и это ограничение записываются в provenance.
- URL БД, Qdrant endpoint и API key не попадают в baseline metadata.
