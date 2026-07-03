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


def retrieval_frequencies(hits_per_query: list[list[str]], corpus_ids: list[str]) -> dict[str, int]:
    """
    Сколько раз каждый чанк корпуса попал в top-k по всем запросам.

    hits_per_query: список top-k (id чанков) на каждый запрос.
    corpus_ids: все id корпуса (включая ни разу не найденные).
    """
    freqs = {cid: 0 for cid in corpus_ids}
    for hits in hits_per_query:
        for cid in hits:
            if cid in freqs:
                freqs[cid] += 1
    return freqs


def coverage(freqs: dict[str, int]) -> float:
    """
    Coverage % — доля корпуса, найденная хотя бы раз.
    1.0 = все чанки находятся; 0.5 = половина ни разу не нашлась.
    """
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
    vals = sorted(freqs.values(), reverse=True)
    total = sum(vals)
    if total == 0:
        return 0.0
    return sum(vals[:top_n]) / total


def hub_leaderboard(freqs: dict[str, int], top_n: int = 10) -> list[tuple[str, int]]:
    """Top-N хабов по частоте попадания в top-k."""
    return sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


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

    def __str__(self) -> str:
        lines = [
            "=" * 64,
            "RETRIEVAL FAIRNESS — отчёт exposure",
            "=" * 64,
            f"  Корпус:   {self.n_chunks} чанков",
            f"  Запросов: {self.n_queries}",
            f"  top-k:    {self.top_k}",
            "-" * 64,
            f"  Coverage:     {self.coverage_pct*100:6.2f}%   (доля корпуса, что находится)",
            f"  Dark matter:  {self.dark_matter_pct*100:6.2f}%   (доля, что НИ РАЗУ не нашлась)",
            f"  Gini:         {self.gini:.3f}   (0=равномерно, 1=концентрация)",
            f"  Hub capture:  top5={self.hub_capture_top5*100:5.1f}%  top10={self.hub_capture_top10*100:5.1f}%",
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
        }


def build_report(
    freqs: dict[str, int],
    n_queries: int,
    top_k: int,
    leaderboard_n: int = 10,
) -> FairnessReport:
    """Собрать сводный FairnessReport из retrieval-frequency."""
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
