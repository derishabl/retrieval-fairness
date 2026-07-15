"""
dashboard.py — HTML-дашборд exposure-смещения.

Генерирует автономный .html (без внешних JS/CSS) с:
- сводными метриками (coverage, dark-matter, Gini, hub-capture)
- Lorenz curve (inline SVG)
- retrieval-frequency histogram (inline SVG, log y)
- hub leaderboard (таблица)
- dark-matter список
- опционально: 2D-проекция корпуса (PCA), dark-matter vs found — разными цветами

Запуск:
  python -m retrieval_fairness probe --corpus c.jsonl --queries q.jsonl --html report.html
  python -m retrieval_fairness dashboard --baseline b.json --html report.html
"""

from __future__ import annotations

import html
import json

from retrieval_fairness.metrics import hub_leaderboard, lorenz
from retrieval_fairness.probe import ProbeResult
from retrieval_fairness.serialize import load_probe
from retrieval_fairness.validation import require_positive_int, validate_vector


def _svg_lorenz(points: list[tuple[float, float]], width: int = 400, height: int = 400) -> str:
    """Lorenz curve как inline SVG. Диагональ = равенство."""
    pad = 30
    w, h = width - 2 * pad, height - 2 * pad
    # кривая (диагональ равенства рисуется ниже отдельным <line>)
    if not points:
        curve = ""
    else:
        pts = []
        for x, y in points:
            px = pad + x * w
            py = (height - pad) - y * h
            pts.append(f"{px:.1f} {py:.1f}")
        curve = "M " + " L ".join(pts)
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="{pad}" y="{pad}" width="{w}" height="{h}" fill="#fafafa" stroke="#ddd"/>
  <line x1="{pad}" y1="{height - pad}" x2="{pad + w}" y2="{pad}" stroke="#ccc" stroke-dasharray="4 3"/>
  <path d="{curve}" fill="none" stroke="#d62728" stroke-width="2"/>
  <text x="{pad}" y="{height - 8}" font-size="11" fill="#666">доля чанков (бедные → богатые)</text>
  <text x="6" y="{pad + 10}" font-size="11" fill="#666" transform="rotate(-90 12 {pad + 30})">доля exposure</text>
  <text x="{width // 2 - 30}" y="18" font-size="12" fill="#333">Lorenz curve</text>
</svg>"""


def _svg_histogram(freqs: dict[str, int], bins: int = 20, width: int = 400, height: int = 200) -> str:
    """Retrieval-frequency histogram (log y)."""
    vals = list(freqs.values())
    if not vals:
        return "<p>нет данных</p>"
    max_v = max(vals)
    if max_v == 0:
        max_v = 1
    # гистограмма по частотам
    counts = [0] * (bins)
    for v in vals:
        idx = min(int(v / max_v * bins), bins - 1)
        counts[idx] += 1
    # log
    import math

    log_counts = [math.log10(c + 1) for c in counts]
    max_lc = max(log_counts) or 1
    pad = 30
    bw = (width - 2 * pad) / bins
    bars = []
    for i, lc in enumerate(log_counts):
        bh = (lc / max_lc) * (height - 2 * pad)
        x = pad + i * bw
        y = (height - pad) - bh
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 1:.1f}" height="{bh:.1f}" fill="#1f77b4"/>')
    bars_str = "\n".join(bars)
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <line x1="{pad}" y1="{height - pad}" x2="{pad + bins * bw}" y2="{height - pad}" stroke="#333"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#333"/>
  {bars_str}
  <text x="{width // 2 - 50}" y="18" font-size="12" fill="#333">Retrieval frequency histogram (log y)</text>
  <text x="{pad}" y="{height - 8}" font-size="10" fill="#666">0</text>
  <text x="{pad + bins * bw - 20}" y="{height - 8}" font-size="10" fill="#666">{max_v}</text>
</svg>"""


def _pca_2d(
    chunks_vectors: list[list[float]],
    labels: list[str],
    freqs: dict[str, int],
    width: int = 480,
    height: int = 360,
    max_pca_points: int = 5000,
) -> str:
    """2D PCA projection with deterministic, dark-matter-first sampling."""
    import random

    import numpy as np
    from sklearn.decomposition import PCA

    require_positive_int(max_pca_points, "max_pca_points")
    if len(chunks_vectors) != len(labels):
        raise ValueError(f"len(chunks_vectors)={len(chunks_vectors)} != len(chunk_ids)={len(labels)}")
    total = len(chunks_vectors)
    if total == 0:
        return "<p>нет векторов для проекции — displayed 0 of 0 chunks</p>"
    dimension = len(chunks_vectors[0])
    for index, vector in enumerate(chunks_vectors):
        validate_vector(vector, name=f"chunks_vectors[{index}]", dim=dimension)
    if total < 2:
        return f"<p>для PCA нужно минимум 2 чанка — displayed {total} of {total} chunks</p>"
    if dimension < 2:
        return f"<p>размерность &lt; 2, проекция невозможна — displayed {total} of {total} chunks</p>"

    dark_indices = [i for i, label in enumerate(labels) if freqs.get(label, 0) == 0]
    found_indices = [i for i, label in enumerate(labels) if freqs.get(label, 0) > 0]
    if total > max_pca_points:
        if len(dark_indices) >= max_pca_points:
            selected = dark_indices[:max_pca_points]
        else:
            rng = random.Random(0)
            remaining = max_pca_points - len(dark_indices)
            selected_found = rng.sample(found_indices, min(remaining, len(found_indices)))
            selected = dark_indices + sorted(selected_found)
    else:
        selected = list(range(total))

    if len(selected) < 2:
        return (
            f"<p>для PCA нужно минимум 2 отображаемых чанка — displayed {len(selected)} of {total} chunks</p>"
        )
    sampled_vectors = [chunks_vectors[i] for i in selected]
    sampled_labels = [labels[i] for i in selected]
    matrix = np.asarray(sampled_vectors, dtype=float)
    projection = PCA(n_components=2).fit_transform(matrix)
    xs = (projection[:, 0] - projection[:, 0].min()) / (np.ptp(projection[:, 0]) + 1e-9)
    ys = (projection[:, 1] - projection[:, 1].min()) / (np.ptp(projection[:, 1]) + 1e-9)
    pad = 20
    points = []
    for index, chunk_id in enumerate(sampled_labels):
        found = freqs.get(chunk_id, 0) > 0
        color = "#1f77b4" if found else "#d62728"
        cx = pad + xs[index] * (width - 2 * pad)
        cy = pad + ys[index] * (height - 2 * pad)
        points.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color}" '
            f'opacity="0.7"><title>{html.escape(chunk_id)}</title></circle>'
        )
    points_html = "\n".join(points)
    return f"""<p>displayed {len(selected)} of {total} chunks</p>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="#fafafa" stroke="#ddd"/>
  {points_html}
  <text x="{width // 2 - 60}" y="18" font-size="12" fill="#333">Corpus map (PCA 2D): blue=found, red=dark-matter</text>
</svg>"""


def build_html(
    result: ProbeResult,
    chunks_vectors: list[list[float]] | None = None,
    chunk_ids: list[str] | None = None,
    title: str = "Retrieval Fairness Report",
    max_pca_points: int = 5000,
) -> str:
    """Сгенерировать автономный HTML-отчёт."""
    if result.report is None:
        raise ValueError("ProbeResult.report is required for a dashboard")
    rep = result.report
    freqs = result.freqs
    lz = lorenz(freqs)
    hist = _svg_histogram(freqs)
    lz_svg = _svg_lorenz(lz)

    # hub leaderboard таблица
    hubs = hub_leaderboard(freqs, top_n=15)
    rows = "\n".join(f"<tr><td>{html.escape(cid)}</td><td align='right'>{cnt}</td></tr>" for cid, cnt in hubs)

    # dark matter
    dm = rep.dark_matter_ids
    dm_list = ", ".join(html.escape(c) for c in dm[:50]) + (f" … (+{len(dm) - 50})" if len(dm) > 50 else "")

    # PCA проекция, если даны векторы
    pca_svg = ""
    if chunks_vectors is not None or chunk_ids is not None:
        if chunks_vectors is None or chunk_ids is None:
            raise ValueError("chunks_vectors and chunk_ids must be provided together")
        pca_svg = _pca_2d(chunks_vectors, chunk_ids, freqs, max_pca_points=max_pca_points)

    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 24px; background: #fff; color: #222; }}
  .metric {{ display: inline-block; margin: 8px 16px 8px 0; padding: 10px 14px; background: #f5f5f5; border-radius: 6px; }}
  .metric .v {{ font-size: 22px; font-weight: 600; }}
  .metric .l {{ font-size: 11px; color: #666; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  td, th {{ border: 1px solid #ddd; padding: 4px 8px; }}
  .dm {{ font-size: 12px; color: #555; max-width: 600px; word-break: break-all; }}
  h2 {{ margin-top: 28px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Корпус: <b>{rep.n_chunks}</b> чанков · Запросов: <b>{rep.n_queries}</b> · top-k: <b>{rep.top_k}</b></p>

<h2>Метрики</h2>
<div>
  <div class="metric"><div class="v">{rep.coverage_pct * 100:.1f}%</div><div class="l">Coverage (of corpus)</div></div>
  <div class="metric"><div class="v">{rep.coverage_of_ceiling * 100:.1f}%</div><div class="l">Of reachable ceiling ({rep.reachability_ceiling} chunks)</div></div>
  <div class="metric"><div class="v">{rep.dark_matter_pct * 100:.1f}%</div><div class="l">Dark matter</div></div>
  <div class="metric"><div class="v">{rep.gini:.3f}</div><div class="l">Gini (0=равномерно)</div></div>
  <div class="metric"><div class="v">{rep.hub_capture_top5 * 100:.1f}%</div><div class="l">Hub capture top5</div></div>
  <div class="metric"><div class="v">{rep.hub_capture_top10 * 100:.1f}%</div><div class="l">Hub capture top10</div></div>
</div>

<div class="row">
  <div>{lz_svg}</div>
  <div>{hist}</div>
</div>

{f"<h2>Карта корпуса (PCA 2D)</h2><div>{pca_svg}</div>" if pca_svg else ""}

<h2>Top хабы</h2>
<table><tr><th>chunk id</th><th>попаданий в top-k</th></tr>
{rows}
</table>

<h2>Dark matter ({len(dm)} чанков)</h2>
<p class="dm">{dm_list or "нет — все чанки находятся"}</p>

<hr><p style="color:#999;font-size:11px">retrieval-fairness · автономный отчёт</p>
</body></html>"""


def render_dashboard(
    result: ProbeResult,
    out_path: str,
    chunks_vectors: list[list[float]] | None = None,
    chunk_ids: list[str] | None = None,
    max_pca_points: int = 5000,
) -> None:
    """Собрать и записать HTML-дашборд в файл."""
    html_str = build_html(
        result,
        chunks_vectors=chunks_vectors,
        chunk_ids=chunk_ids,
        max_pca_points=max_pca_points,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)


def render_dashboard_from_baseline(baseline_path: str, out_path: str, corpus_path: str | None = None) -> None:
    """
    Собрать дашборд из сохранённого baseline JSON.
    Если дан corpus_path (JSONL с векторами) — добавит PCA-проекцию.
    """
    result = load_probe(baseline_path)
    chunks_vectors = None
    chunk_ids = None
    if corpus_path:
        rows = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        chunk_ids = [r["id"] for r in rows]
        chunks_vectors = [r["vector"] for r in rows]
    render_dashboard(result, out_path, chunks_vectors=chunks_vectors, chunk_ids=chunk_ids)
