"""
demo.py — демонстрация retrieval_fairness на синтетическом корпусе.

Корпус: 30 чанков по 3 темы (отпуск, VPN, зарплата) + 1 «хаб»,
семантически близкий ко всем темам сразу (имитирует super-hub).
Запросы: по теме + общие. TF-IDF как лёгкий эмбеддер (без тяжёлых моделей).

Демо показывает: hub capture (хаб захватывает выдачу) и dark matter
(узкие чанки не находятся общими запросами).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval_fairness.adapters import InMemoryVectorStore
from retrieval_fairness.probe import probe
from retrieval_fairness.types import Chunk, Query


def _build_corpus() -> tuple[list[Chunk], TfidfVectorizer]:
    topics = {
        "отпуск": [
            "Отпуск оформляется через HR-портал за две недели до даты.",
            "Ежегодный оплачиваемый отпуск составляет 28 календарных дней.",
            "Заявление на отпуск подаётся в личном кабинете сотрудника.",
            "Перенос отпуска согласуется с руководителем и HR.",
            "Компенсация неиспользованного отпуска выплачивается при увольнении.",
            "Отпуск в первый год работы предоставляется через шесть месяцев.",
            "Разделение отпуска на части допускается, одна часть не менее 14 дней.",
            "Декретный отпуск оформляется через больничный лист и HR.",
            "Отпуск без сохранения зарплаты согласовывается индивидуально.",
            "График отпусков утверждается в декабре на следующий год.",
        ],
        "vpn": [
            "Для входа в VPN используйте корпоративное приложение.",
            "Одноразовый код для VPN получают из аутентификатора.",
            "Настройка VPN на телефоне описана в IT-инструкции.",
            "VPN-подключение из-за границы требует заявки в безопасниках.",
            "Сбой VPN регистрируется в сервис-деске с указанием времени.",
            "Срок действия VPN-сертификата — один год, продлевается автоматически.",
            "Двухфакторная аутентификация обязательна для VPN.",
            "Лог VPN-подключений хранится 90 дней в системе аудита.",
            "Отзыв VPN-доступа при увольнении выполняет администратор.",
            "VPN-профиль скачивается из личного кабинета в разделе доступы.",
        ],
        "зарплата": [
            "Зарплата выплачивается двумя частями: аванс и остаток.",
            "Расчётный лист доступен в личном кабинете после выплаты.",
            "Премия за квартал начисляется по итогам KPI.",
            "НДФЛ удерживается из зарплаты автоматически.",
            "Больничный оплачивается по среднему заработку за два года.",
            "Реквизиты карты для зарплаты передаются в бухгалтерию при найме.",
            "Переработки компенсируются отгулом или доплатой по заявлению.",
            "Справка о доходах выдаётся бухгалтерией по запросу.",
            "Аванс выплачивается 20-го, остаток — 5-го числа.",
            "Удержания из зарплаты отражаются в расчётном листе.",
        ],
    }

    texts: list[str] = []
    ids: list[str] = []
    for topic, docs in topics.items():
        for i, t in enumerate(docs):
            ids.append(f"{topic}_{i}")
            texts.append(t)

    # hub: намеренно «липнет» ко всем темам (имитация super-hub)
    ids.append("HUB_universal")
    texts.append("Сводный регламент по отпуску, VPN и зарплате: единый раздел для сотрудника.")

    vec = TfidfVectorizer()
    mat = vec.fit_transform(texts).toarray().astype(float)
    chunks = [Chunk(id=i, text=t, vector=v.tolist()) for i, t, v in zip(ids, texts, mat)]
    return chunks, vec


def _build_queries(vec: TfidfVectorizer) -> list[Query]:
    query_texts = [
        ("q_leave_1", "как оформить отпуск"),
        ("q_leave_2", "сколько дней отпуска положено"),
        ("q_leave_3", "перенос отпуска"),
        ("q_vpn_1", "настроить VPN на телефоне"),
        ("q_vpn_2", "код для VPN входа"),
        ("q_vpn_3", "VPN из-за границы"),
        ("q_salary_1", "когда аванс и зарплата"),
        ("q_salary_2", "расчётный лист"),
        ("q_salary_3", "премия за квартал"),
        ("q_general_1", "общий регламент сотрудника"),
        ("q_general_2", "инструкции для новичка"),
    ]
    out = []
    for qid, t in query_texts:
        v = vec.transform([t]).toarray()[0].astype(float)
        out.append(Query(id=qid, vector=v.tolist(), text=t))
    return out


def run_demo(top_k: int = 5) -> None:
    chunks, vec = _build_corpus()
    queries = _build_queries(vec)
    store = InMemoryVectorStore(chunks)

    result = probe(store, queries, top_k=top_k)
    if result.report is None:
        raise RuntimeError("probe returned no report")
    print(result.report)

    # покажем dark matter явно
    dm = result.report.dark_matter_ids
    if dm:
        print(f"\nDark matter — чанки, которые ни разу не нашлись ({len(dm)}):")
        for cid in dm:
            txt = next((c.text for c in chunks if c.id == cid), "?")
            print(f"  {cid:30} {txt[:60]}")

    # покажем, что хаб доминирует
    hub_in_top = sum(1 for hits in result.hits_per_query if "HUB_universal" in hits)
    print(f"\nХаб 'HUB_universal' попал в top-{top_k} в {hub_in_top}/{len(queries)} запросах")
    print("  → иллюстрация hub capture: общий чанк вытесняет узкотематичные")


def _rebuild_with_new_embedder(chunks: list[Chunk], queries: list[Query]):
    """
    Имитация миграции эмбеддера: новый TfidfVectorizer с другим ngram_range
    поверх ТЕХ ЖЕ текстов. Возвращает новые chunks и queries с пересчитанными
    векторами. Имитирует «сменили эмбеддер — что поменялось в retrieval».
    """
    texts = [c.text for c in chunks]
    query_texts = [q.text for q in queries]
    new_vec = TfidfVectorizer(ngram_range=(1, 2))  # было (1,1) по умолчанию
    all_texts = texts + query_texts
    new_vec.fit(all_texts)
    new_chunk_vecs = new_vec.transform(texts).toarray().astype(float)
    new_query_vecs = new_vec.transform(query_texts).toarray().astype(float)
    new_chunks = [Chunk(id=c.id, text=c.text, vector=v.tolist()) for c, v in zip(chunks, new_chunk_vecs)]
    new_queries = [Query(id=q.id, vector=v.tolist(), text=q.text) for q, v in zip(queries, new_query_vecs)]
    return new_chunks, new_queries


def run_migration_diff_demo(top_k: int = 5) -> None:
    """Демонстрация regression diff: baseline (TF-IDF 1-gram) vs new (1-2 gram)."""
    from retrieval_fairness.diff import diff_reports

    chunks, vec = _build_corpus()
    queries = _build_queries(vec)
    baseline = probe(InMemoryVectorStore(chunks), queries, top_k=top_k)

    new_chunks, new_queries = _rebuild_with_new_embedder(chunks, queries)
    candidate = probe(InMemoryVectorStore(new_chunks), new_queries, top_k=top_k)

    print("=" * 64)
    print("ДЕМО: regression diff при миграции эмбеддера (1-gram -> 1,2-gram)")
    print("=" * 64)
    print("\n--- BASELINE ---")
    print(baseline.report)
    print("\n--- CANDIDATE ---")
    print(candidate.report)
    print()
    d = diff_reports(baseline, candidate)
    print(d)
