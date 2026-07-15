"""Exposure concentration and reachability metrics for retrieval workloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval_fairness.validation import (
    require_non_negative_int,
    require_positive_int,
    validate_unique_ids,
)


@dataclass(frozen=True)
class FrequencyStats:
    """Immutable calculation context shared by compound report metrics."""

    items: tuple[tuple[str, int], ...]
    values: tuple[int, ...]
    sorted_values: tuple[int, ...]
    total: int
    found: int

    @classmethod
    def from_frequencies(cls, freqs: dict[str, int]) -> FrequencyStats:
        validate_unique_ids(list(freqs), name="frequency IDs")
        items = tuple(freqs.items())
        for chunk_id, value in items:
            require_non_negative_int(value, f"frequency[{chunk_id!r}]")
        values = tuple(value for _, value in items)
        return cls(
            items=items,
            values=values,
            sorted_values=tuple(sorted(values)),
            total=sum(values),
            found=sum(value > 0 for value in values),
        )

    @property
    def n(self) -> int:
        return len(self.values)


def _stats(freqs: dict[str, int]) -> FrequencyStats:
    return FrequencyStats.from_frequencies(freqs)


def retrieval_frequencies(hits_per_query: list[list[str]], corpus_ids: list[str]) -> dict[str, int]:
    """Count how often each corpus ID occurs in the workload's top-k rows."""
    validate_unique_ids(corpus_ids, name="corpus_ids")
    freqs = dict.fromkeys(corpus_ids, 0)
    for query_index, hits in enumerate(hits_per_query):
        validate_unique_ids(hits, name=f"hits_per_query[{query_index}]")
        for chunk_id in hits:
            if chunk_id not in freqs:
                raise ValueError(f"hit ID {chunk_id!r} is not present in corpus_ids")
            freqs[chunk_id] += 1
    return freqs


def _coverage(stats: FrequencyStats) -> float:
    return stats.found / stats.n if stats.n else 0.0


def coverage(freqs: dict[str, int]) -> float:
    """Share of corpus chunks retrieved at least once."""
    return _coverage(_stats(freqs))


def dark_matter(freqs: dict[str, int]) -> float:
    """Share of corpus chunks never retrieved by the workload."""
    stats = _stats(freqs)
    return 1.0 - _coverage(stats) if stats.n else 0.0


def _gini(stats: FrequencyStats) -> float:
    if stats.n == 0 or stats.total == 0:
        return 0.0
    weighted = sum(index * value for index, value in enumerate(stats.sorted_values, start=1))
    result = (2 * weighted) / (stats.n * stats.total) - (stats.n + 1) / stats.n
    return max(0.0, min(1.0, result))


def gini(freqs: dict[str, int]) -> float:
    """Gini coefficient of exposure frequencies (0 uniform, 1 concentrated)."""
    return _gini(_stats(freqs))


def _lorenz(stats: FrequencyStats) -> list[tuple[float, float]]:
    if stats.n == 0:
        return []
    if stats.total == 0:
        return [(index / stats.n, 0.0) for index in range(stats.n + 1)]
    points = [(0.0, 0.0)]
    cumulative = 0
    for index, value in enumerate(stats.sorted_values, start=1):
        cumulative += value
        points.append((index / stats.n, cumulative / stats.total))
    return points


def lorenz(freqs: dict[str, int]) -> list[tuple[float, float]]:
    """Lorenz curve from (0, 0) to (1, 1), ordered by exposure."""
    return _lorenz(_stats(freqs))


def downsample_lorenz(freqs: dict[str, int], max_points: int = 512) -> list[tuple[float, float]]:
    """Return deterministic quantile points without materializing a full curve."""
    require_positive_int(max_points, "max_points")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    stats = _stats(freqs)
    total_points = stats.n + 1 if stats.n else 0
    if total_points <= max_points:
        return _lorenz(stats)
    indices = sorted({round(position * stats.n / (max_points - 1)) for position in range(max_points)})
    wanted = set(indices)
    points: list[tuple[float, float]] = []
    cumulative = 0
    if 0 in wanted:
        points.append((0.0, 0.0))
    for index, value in enumerate(stats.sorted_values, start=1):
        cumulative += value
        if index in wanted:
            y = cumulative / stats.total if stats.total else 0.0
            points.append((index / stats.n, y))
    return points


def _hub_capture(stats: FrequencyStats, top_n: int) -> float:
    if stats.total == 0:
        return 0.0
    return sum(stats.sorted_values[-top_n:]) / stats.total


def hub_capture(freqs: dict[str, int], top_n: int = 5) -> float:
    """Share of total exposure captured by the top-N chunks."""
    require_positive_int(top_n, "top_n")
    return _hub_capture(_stats(freqs), top_n)


def hub_leaderboard(freqs: dict[str, int], top_n: int = 10) -> list[tuple[str, int]]:
    """Top-N chunks with a deterministic ID tie-break."""
    require_non_negative_int(top_n, "top_n")
    stats = _stats(freqs)
    return sorted(stats.items, key=lambda item: (-item[1], item[0]))[:top_n]


def reachability_ceiling(n_chunks: int, n_queries: int, top_k: int) -> int:
    """Maximum distinct chunks reachable by ``n_queries`` top-k result rows."""
    require_non_negative_int(n_chunks, "n_chunks")
    require_non_negative_int(n_queries, "n_queries")
    require_positive_int(top_k, "top_k")
    return min(n_chunks, n_queries * top_k)


@dataclass
class FairnessReport:
    """Derived exposure report. Raw observations remain the source of truth."""

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
    found_count: int | None = None
    dark_matter_count_exact: int | None = None

    @property
    def reachability_ceiling(self) -> int:
        return reachability_ceiling(self.n_chunks, self.n_queries, self.top_k)

    @property
    def dark_matter_count(self) -> int:
        if self.dark_matter_count_exact is not None:
            return self.dark_matter_count_exact
        return len(self.dark_matter_ids)

    @property
    def coverage_of_ceiling(self) -> float:
        ceiling = self.reachability_ceiling
        if ceiling == 0:
            return 0.0
        found = self.found_count
        if found is None:
            if self.dark_matter_ids or self.coverage_pct >= 1.0:
                found = self.n_chunks - len(self.dark_matter_ids)
            else:
                found = round(self.coverage_pct * self.n_chunks)
        return min(1.0, found / ceiling)

    @property
    def lorenz_points_total(self) -> int:
        return self.n_chunks + 1 if self.n_chunks else 0

    def __str__(self) -> str:
        lines = [
            "=" * 64,
            "RETRIEVAL FAIRNESS — exposure report",
            "=" * 64,
            f"  Corpus:  {self.n_chunks} chunks",
            f"  Queries: {self.n_queries}",
            f"  top-k:   {self.top_k}",
            "-" * 64,
            f"  Coverage:       {self.coverage_pct * 100:6.2f}%",
            f"  Of reachable:   {self.coverage_of_ceiling * 100:6.2f}% (ceiling {self.reachability_ceiling})",
            f"  Dark matter:    {self.dark_matter_pct * 100:6.2f}%",
            f"  Gini:           {self.gini:.3f}",
            f"  Hub capture:    top5={self.hub_capture_top5 * 100:5.1f}% "
            f"top10={self.hub_capture_top10 * 100:5.1f}%",
            "-" * 64,
            "  Top hubs (id: top-k occurrences):",
        ]
        lines.extend(f"    {chunk_id:30} {count}" for chunk_id, count in self.hub_leaderboard)
        lines.extend(
            [
                "-" * 64,
                f"  Dark matter: {self.dark_matter_count} chunks never retrieved",
                "=" * 64,
            ]
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
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
            "dark_matter_count": self.dark_matter_count,
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
    *,
    detail: str = "full",
) -> FairnessReport:
    """Build a report with one validation/sort pass.

    ``detail='summary'`` omits O(n) Lorenz and dark-ID collections while
    retaining exact counts and metrics.
    """
    require_non_negative_int(n_queries, "n_queries")
    require_positive_int(top_k, "top_k")
    require_non_negative_int(leaderboard_n, "leaderboard_n")
    if detail not in {"full", "summary"}:
        raise ValueError("detail must be 'full' or 'summary'")

    stats = _stats(freqs)
    dark_count = stats.n - stats.found
    dark_ids = [chunk_id for chunk_id, value in stats.items if value == 0] if detail == "full" else []
    leaderboard = sorted(stats.items, key=lambda item: (-item[1], item[0]))[:leaderboard_n]
    coverage_value = _coverage(stats)
    return FairnessReport(
        n_chunks=stats.n,
        n_queries=n_queries,
        top_k=top_k,
        coverage_pct=coverage_value,
        dark_matter_pct=(1.0 - coverage_value if stats.n else 0.0),
        gini=_gini(stats),
        hub_capture_top5=_hub_capture(stats, 5),
        hub_capture_top10=_hub_capture(stats, 10),
        hub_leaderboard=leaderboard,
        lorenz_curve=_lorenz(stats) if detail == "full" else [],
        dark_matter_ids=dark_ids,
        found_count=stats.found,
        dark_matter_count_exact=dark_count,
    )
