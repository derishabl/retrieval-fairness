# retrieval-fairness vs существующие — честная сравнительная таблица

Цель: чётко показать, где мы сильнее, а где — нет. Без преувеличений.

## Позиционирование

retrieval-fairness — **exposure-bias audit для векторного поиска/RAG**:
показывает, какую долю корпуса не находят запросы, меряет концентрацию
(Gini), захват хабами, и даёт CI-гейт/regression-diff для миграций
эмбеддера/чанкинга. Метрики заимствованы (честно: Gini/Lorenz — экономика,
retrievability — T-Retrievability); новизна — packaging под RAG.

## Сравнение

| Инструмент | Фокус | Exposure bias? | CI-гейт? | Regression diff? | Dashboard? | Векторный стор | Зрелость |
|---|---|---|---|---|---|---|---|
| **retrieval-fairness** | exposure audit RAG | ✅ ядро | ✅ | ✅ | ✅ | adapter (FAISS/Qdrant/...) | ранний |
| T-Retrievability | retrievability метрика | ✅ (метрика) | ❌ | ❌ | ❌ | TREC run-files | статья+код |
| semantic-coverage | knowledge gaps RAG | частично | ❌ | ❌ | ❌ | vector store | мелкий |
| rag-sentinel | governance RAG | ❌ (freshness/PII) | ❌ | частично | частично | vector store | мелкий |
| RankAudit | generic ranking fairness | ✅ (general) | ❌ | ❌ | ❌ | black-box ranker | мелкий |
| LongProbe / vector-guardrails | retrieval regression | ❌ | ✅ | ✅ | ❌ | RAG pipelines | мелкий |
| Drift-Adapter | embedding migration | ❌ | ❌ | ❌ | ❌ | research | статья |

## Где мы сильнее

1. **Фокус именно на exposure bias + dark matter** — ни у кого это не ядро.
   semantic-coverage ближе всех, но про knowledge gaps, не про «какая
   доля не находится».
2. **Связка exposure-метрик + CI-гейт + regression-diff + dashboard в одном** —
   фрагментированная ниша: regression есть у LongProbe, гейта нет;
   метрики есть у T-Retrievability, продукта нет.
3. **Adapter-контракт под векторные БД** — T-Retrievability работает с
   TREC run-files; мы — с реальными векторными сторами.

## Где мы слабее (честно)

1. **Метрики не наши** — собраны из IR-fairness/T-Retrievability. Любой
   может собрать аналог за неделю. Defensibility = execution, не патент.
2. **Нет query-логов у ранних клиентов** — синтетика (synth) приближение,
   слабее реальной нагрузки.
3. **Один адаптер сейчас (InMemory)** — FAISS/Qdrant/Pinecone в roadmap.
4. **Без LLM-парафраза в synth** — детерминированный TF-IDF, грубее.
5. **Малый корпус в демо** — нужен кейс на публичном корпусе (NQ/MS MARCO)
   с реальными цифрами exposure.

## Дифференциаторы, которые надо усилить

- **CI-гейт как killer-feature**: «блокируй деплой при падении coverage» —
  этого нет ни у кого в exposure-домене. Ставка на dev-ops интеграцию.
- **Dark-matter визуализация (PCA-карта)**: показывает *какие* темы
  отрезаны, не только число. Конкуренты дают числа, не картинку.
- **Synth без логов**: снижает порог входа — команда без query-логов всё
  равно может запустить аудит.

## Чего не делать (чтобы не размывать)

- Не лезть в freshness/PII/governance — это rag-sentinel, не наша ниша.
- Не делать generic ranking fairness — это RankAudit.
- Не делать attack-cost/poisoning — перегретая отдельная область (см. archive/).
