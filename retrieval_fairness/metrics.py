"""
metrics.py — метрики exposure-смещения векторного поиска.

Ядро честно заимствовано из IR-fairness / экономики:
  - Gini, Lorenz — классика измерения концентрации
  - retrievability — из T-Retrievability (ACM DOI 10.1145/3746252.3760820)
Product-упаковка (coverage %, dark-matter %, hub-capture ratio) — наша.

Все метрики считаются по retrieval-frequency: сколько раз каждый чанк
попал в top-k по запросам workload'а. Источник частот — probe.probe().
"""

from __future__ import annotations
from dataclasses import dataclass, field

from retrieval_fairness.validation import (
    require_non_negative_int,
    require_positive_int,
    validate_unique_ids,
)


def _validate_freqs(freqs: dict[str, int]) -> None:
    validate_unique_ids(list(freqs), name="frequency IDs")
    for chunk_id, value in freqs.items():
        require_non_negative_int(value, f"frequency[{chunk_id!r}]")


def retrieval_frequencies(hits_per_query: list[list[str]], corpus_ids: list[str]) -> dict[str, int]:
    """
    Сколько раз каждый чанк корпуса попал в top-k по всем запросам.

    hits_per_query: список top-k (id чанков) на каждый запрос.
    corpus_ids: все id корпуса (включая ни разу не найденные).
    """
    validate_unique_ids(corpus_ids, name="corpus_ids")
    freqs = {cid: 0 for cid in corpus_ids}
    for query_index, hits in enumerate(hits_per_query):
        validate_unique_ids(hits, name=f"hits_per_query[{query_index}]")
        for cid in hits:
            if cid not in freqs:
                raise ValueError(f"hit ID {cid!r} is not present in corpus_ids")
            freqs[cid] += 1
    return freqs


def coverage(freqs: dict[str, int]) -> float:
    """
    Coverage % — доля корпуса, найденная хотя бы раз.
    1.0 = все чанки находятся; 0.5 = половина ни разу не нашлась.
    """
    _validate_freqs(freqs)
    if not freqs:
        return 0.0
    found = sum(1 for v in freqs.values() if v > 0)
    return found / len(freqs)


def dark_matter(freqs: dict[str, int]) -> float:
    """
    Dark-matter % — доля корпуса, НИ РАЗУ не найденная ни одним запросом.
    Дополнение coverage до 1: dark_matter = 1 - coverage.
    Для пустого корпуса — 0 (нечего быть «тёмным»).
    """
    if not freqs:
        return 0.0
    return 1.0 - coverage(freqs)


def gini(freqs: dict[str, int]) -> float:
    """
    Gini коэффициент концентрации exposure.
    0 = равномерно (все чанки находятся одинаково часто);
    1 = максимально неравномерно (всё в одном чанке).

    Формула: G = (Σ_i Σ_j |x_i - x_j|) / (2 * n * Σ x_i).
    Заимствовано из экономики / IR-fairness.
    """
    _validate_freqs(freqs)
    vals = list(freqs.values())
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0  # ничего не находится — концентрация не определена, считаем 0
    # стабильная формула через отсортированные значения
    vals_sorted = sorted(vals)
    cum = 0.0
    for i, v in enumerate(vals_sorted, start=1):
        cum += i * v
    g = (2 * cum) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, g))


def lorenz(freqs: dict[str, int]) -> list[tuple[float, float]]:
    """
    Lorenz curve: точки (доля чанков, доля накопленного exposure),
    от беднейших к богатейшим. (0,0) ... (1,1).
    Диагональ = равенство; провисание = неравенство.
    """
    _validate_freqs(freqs)
    vals = sorted(freqs.values())
    n = len(vals)
    if n == 0:
        return []
    total = sum(vals)
    if total == 0:
        return [(i / n, 0.0) for i in range(n + 1)]
    points = [(0.0, 0.0)]
    cum = 0
    for i, v in enumerate(vals, start=1):
        cum += v
        points.append((i / n, cum / total))
    return points


def hub_capture(freqs: dict[str, int], top_n: int = 5) -> float:
    """
    Hub-capture ratio — доля всего exposure, приходящаяся на top-N хабов.
    1.0 = всё попадает в N чанков; ~0 = хабов нет.
    """
    require_positive_int(top_n, "top_n")
    _validate_freqs(freqs)
    vals = sorted(freqs.values(), reverse=True)
    total = sum(vals)
    if total == 0:
        return 0.0
    return sum(vals[:top_n]) / total


def hub_leaderboard(freqs: dict[str, int], top_n: int = 10) -> list[tuple[str, int]]:
    """Top-N хабов по частоте попадания в top-k."""
    require_non_negative_int(top_n, "top_n")
    _validate_freqs(freqs)
    return sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def reachability_ceiling(n_chunks: int, n_queries: int, top_k: int) -> int:
    """
    Workload-потолок coverage: сколько уникальных чанков В ПРИНЦИПЕ
    может быть найдено данным workload'ом. Без него нельзя трактовать
    coverage на больших корпусах: 3452 запроса × top-10 = 34520
    максимально достижимых чанков, поэтому coverage 11.7% на 260k-корпусе —
    это 88% от достижимого, а не «ретривер плохой».

    Потолок = min(n_chunks, n_queries * top_k): нельзя найти больше
    уникальных чанков, чем n_queries*top_k (по top-k на каждый запрос),
    и не больше, чем есть в корпусе.
    Novelty-в-packaging: формулировка coverage как «% от workload-потолка»
    в продукте не встречается (ни у retobs, ни у EmbedAudit, ни у
    T-Retrievability как продукта); см. docs/case_study_nq.md (полный NQ).
    """
    return min(n_chunks, n_queries * top_k)


@dataclass
class FairnessReport:
    """Сводный отчёт по метрикам exposure."""

    n_chunks: int
    n_queries: int
    top_k: int
    coverage_pct: float
    dark_matter_pct: float
    gini: float
    hub_capture_top5: float
    hub_capture_top10: float
    hub_leaderboard: list[tuple[str, int]] = field(default_factory=list)
    lorenz_curve: list[tuple[float, float]] = field(default_factory=list)
    dark_matter_ids: list[str] = field(default_factory=list)

    @property
    def reachability_ceiling(self) -> int:
        """Workload-потолок: сколько уникальных чанков В ПРИНЦИПЕ достижимо."""
        return reachability_ceiling(self.n_chunks, self.n_queries, self.top_k)

    @property
    def coverage_of_ceiling(self) -> float:
        """
        Coverage как доля ОТ ДОСТИЖИМОГО ПОТОЛКА, а не от всего корпуса.
        1.0 = ретривер исчерпал всё, что workload физически может достать
        (не вина ретривера, что потолок < корпуса). Импользовать вместе
        с coverage_pct: последний — «процент корпуса», этот — «насколько
        хорошо отработано в рамках достижимого».
        """
        ceil = self.reachability_ceiling
        if ceil <= 0:
            return 0.0
        # Точный подсчёт через dark_matter_ids (n - dark = found). Реконструкция
        # из округлённого coverage_pct (JSON round(4)) на 260k-корпусе давала бы
        # ошибку до ±13 чанков — fallback только если ids не заполнены.
        if self.dark_matter_ids or self.coverage_pct >= 1.0:
            found = self.n_chunks - len(self.dark_matter_ids)
        else:
            found = round(self.coverage_pct * self.n_chunks)
        return min(1.0, found / ceil)

    def __str__(self) -> str:
        lines = [
            "=" * 64,
            "RETRIEVAL FAIRNESS — отчёт exposure",
            "=" * 64,
            f"  Корпус:   {self.n_chunks} чанков",
            f"  Запросов: {self.n_queries}",
            f"  top-k:    {self.top_k}",
            "-" * 64,
            f"  Coverage:     {self.coverage_pct * 100:6.2f}%   (доля корпуса, что находится)",
            f"  из достижимого: {self.coverage_of_ceiling * 100:6.2f}%   (от workload-потолка {self.reachability_ceiling} чанков)",
            f"  Dark matter:  {self.dark_matter_pct * 100:6.2f}%   (доля, что НИ РАЗУ не нашлась)",
            f"  Gini:         {self.gini:.3f}   (0=равномерно, 1=концентрация)",
            f"  Hub capture:  top5={self.hub_capture_top5 * 100:5.1f}%  top10={self.hub_capture_top10 * 100:5.1f}%",
            "-" * 64,
            "  Top хабы (id: сколько раз в top-k):",
        ]
        for cid, cnt in self.hub_leaderboard:
            lines.append(f"    {cid:30} {cnt}")
        lines.append("-" * 64)
        lines.append(f"  Dark matter: {len(self.dark_matter_ids)} чанков ни разу не найдены")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "n_queries": self.n_queries,
            "top_k": self.top_k,
            "coverage_pct": round(self.coverage_pct, 4),
            "dark_matter_pct": round(self.dark_matter_pct, 4),
            "gini": round(self.gini, 4),
            "hub_capture_top5": round(self.hub_capture_top5, 4),
            "hub_capture_top10": round(self.hub_capture_top10, 4),
            "hub_leaderboard": self.hub_leaderboard,
            "dark_matter_count": len(self.dark_matter_ids),
            "dark_matter_ids": self.dark_matter_ids,
            "lorenz_curve": [[round(x, 6), round(y, 6)] for x, y in self.lorenz_curve],
            "reachability_ceiling": self.reachability_ceiling,
            "coverage_of_ceiling": round(self.coverage_of_ceiling, 4),
        }


def build_report(
    freqs: dict[str, int],
    n_queries: int,
    top_k: int,
    leaderboard_n: int = 10,
) -> FairnessReport:
    """Собрать сводный FairnessReport из retrieval-frequency."""
    _validate_freqs(freqs)
    require_non_negative_int(n_queries, "n_queries")
    require_positive_int(top_k, "top_k")
    require_non_negative_int(leaderboard_n, "leaderboard_n")
    n = len(freqs)
    dm_ids = [cid for cid, v in freqs.items() if v == 0]
    return FairnessReport(
        n_chunks=n,
        n_queries=n_queries,
        top_k=top_k,
        coverage_pct=coverage(freqs),
        dark_matter_pct=dark_matter(freqs),
        gini=gini(freqs),
        hub_capture_top5=hub_capture(freqs, 5),
        hub_capture_top10=hub_capture(freqs, 10),
        hub_leaderboard=hub_leaderboard(freqs, leaderboard_n),
        lorenz_curve=lorenz(freqs),
        dark_matter_ids=dm_ids,
    )
