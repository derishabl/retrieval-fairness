"""
synth.py — antihub self-query аудит: синтетические запросы из корпуса
(для команд без query-логов).

Семантика: для каждого чанка генерируем запрос, который ДОЛЖЕН был бы
его найти. Чанк, который не находится даже нацеленным на него запросом —
невидим ни из какого разумного направления (antihub, см. hubness-
литературу: Radovanović et al., JMLR 2010). Практики рекомендуют такой
self-query аудит на каждом обновлении корпуса.

Exposure зависит от распределения реальных запросов; синтетика — грубый
симулятор workload'а, но точный детектор невидимости.

Подход (без LLM-зависимостей, лёгкий):
  для каждого чанка извлекаем top-N ключевых терминов (TF-IDF веса) и
  собираем «запрос» как конкатенацию этих терминов — эмбеддится тем же
  векторизатором, что и корпус. Это имитирует «запрос, который хотел бы
  найти этот чанк». Дальше probe меряет, какие чанки реально находятся
  такими «целевыми» запросами.

Опционально (future): LLM-парафраз; сейчас — детерминированный TF-IDF.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval_fairness.types import Chunk, Query
from retrieval_fairness.validation import require_positive_int, validate_unique_ids


def _keyword_query(text: str, vec: TfidfVectorizer, n_terms: int = 5) -> str:
    """Извлечь top-N ключевых терминов чанка как псевдо-запрос."""
    vec_text = vec.transform([text])
    if vec_text.nnz == 0:
        return text[:60]
    arr = vec_text.toarray()[0]
    idx = np.argsort(-arr)[:n_terms]
    terms = vec.get_feature_names_out()
    return " ".join(terms[i] for i in idx if arr[i] > 0)


def synth_queries_from_corpus(
    chunks: list[Chunk],
    n_per_chunk: int = 1,
    n_terms: int = 5,
    query_style: str = "keywords",
) -> tuple[list[Query], TfidfVectorizer]:
    """
    Сгенерировать синтетические запросы из корпуса.

    Возвращает (queries, vectorizer) — vectorizer нужно использовать,
    чтобы получить векторы запросов, совместимые с InMemoryVectorStore
    (тот же эмбеддер, что и корпус).

    query_style:
      "keywords" — top-N TF-IDF терминов чанка (по умолчанию)
      "text"     — первые 60 символов чанка (как «цитатный» запрос)
    """
    require_positive_int(n_per_chunk, "n_per_chunk")
    require_positive_int(n_terms, "n_terms")
    if not chunks:
        raise ValueError("chunks must not be empty")
    validate_unique_ids([chunk.id for chunk in chunks], name="corpus IDs")
    if query_style not in {"keywords", "text"}:
        raise ValueError("query_style must be 'keywords' or 'text'")
    texts = [c.text for c in chunks]
    vec = TfidfVectorizer()
    vec.fit(texts)

    queries: list[Query] = []
    for c in chunks:
        for j in range(n_per_chunk):
            q_text = c.text[:60] if query_style == "text" else _keyword_query(c.text, vec, n_terms=n_terms)
            q_vec = vec.transform([q_text]).toarray()[0].astype(float)
            qid = f"synth_{c.id}_{j}"
            queries.append(Query(id=qid, vector=q_vec.tolist(), text=q_text))
    return queries, vec


def synth_probe(
    chunks: list[Chunk],
    top_k: int = 10,
    n_per_chunk: int = 1,
    n_terms: int = 5,
    query_style: str = "keywords",
):
    """
    Полный цикл: сгенерировать синтетические запросы, построить стор,
    прогнать probe. Удобный entrypoint для CLI.

    Возвращает ProbeResult.
    """
    from retrieval_fairness.adapters import InMemoryVectorStore
    from retrieval_fairness.probe import probe

    require_positive_int(top_k, "top_k")
    queries, vec = synth_queries_from_corpus(
        chunks, n_per_chunk=n_per_chunk, n_terms=n_terms, query_style=query_style
    )
    # переэмбеддить чанки тем же векторизатором, чтобы размерности совпадали
    chunk_vecs = vec.transform([c.text for c in chunks]).toarray().astype(float)
    reembedded = [Chunk(id=c.id, text=c.text, vector=v.tolist()) for c, v in zip(chunks, chunk_vecs)]
    store = InMemoryVectorStore(reembedded)
    return probe(store, queries, top_k=top_k, embedder="tfidf", embedder_revision="scikit-learn")
