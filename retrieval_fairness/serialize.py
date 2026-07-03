"""
serialize.py — сохранение/загрузка ProbeResult в JSON.

Нужно для baseline: прогон -> JSON на диске; будущий прогон сравнивается
с загруженным baseline через diff.diff_reports.
"""

from __future__ import annotations
import json

from retrieval_fairness.probe import ProbeResult
from retrieval_fairness.metrics import FairnessReport


def probe_to_json(result: ProbeResult) -> dict:
    """ProbeResult -> машиночитаемый dict (с hits_per_query и freqs)."""
    assert result.report is not None
    return {
        "freqs": result.freqs,
        "hits_per_query": result.hits_per_query,
        "report": result.report.to_dict(),
    }


def save_probe(result: ProbeResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(probe_to_json(result), f, ensure_ascii=False, indent=2)


def load_probe(path: str) -> ProbeResult:
    """Восстановить ProbeResult из JSON. report пересобирается из freqs."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    freqs: dict[str, int] = {k: int(v) for k, v in data["freqs"].items()}
    hits: list[list[str]] = data["hits_per_query"]
    rep_dict = data["report"]
    report = FairnessReport(
        n_chunks=rep_dict["n_chunks"],
        n_queries=rep_dict["n_queries"],
        top_k=rep_dict["top_k"],
        coverage_pct=rep_dict["coverage_pct"],
        dark_matter_pct=rep_dict["dark_matter_pct"],
        gini=rep_dict["gini"],
        hub_capture_top5=rep_dict["hub_capture_top5"],
        hub_capture_top10=rep_dict["hub_capture_top10"],
        hub_leaderboard=[tuple(x) for x in rep_dict.get("hub_leaderboard", [])],
        lorenz_curve=[tuple(p) for p in rep_dict.get("lorenz_curve", [])],
        dark_matter_ids=rep_dict.get(
            "dark_matter_ids", [cid for cid, v in freqs.items() if v == 0]
        ),
    )
    return ProbeResult(freqs=freqs, hits_per_query=hits, report=report)
