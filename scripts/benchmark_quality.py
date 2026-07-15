"""Reproducible 1M-frequency summary benchmark for the quality milestone."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc

from retrieval_fairness.metrics import build_report
from retrieval_fairness.probe import ProbeResult
from retrieval_fairness.serialize import probe_summary_to_json


def run(size: int) -> dict[str, float | int]:
    frequencies = {f"chunk-{index:07d}": (1 if index % 10 == 0 else 0) for index in range(size)}
    hit_ids = [f"chunk-{index:07d}" for index in range(0, size, 10)]
    hits = [[chunk_id] for chunk_id in hit_ids]
    query_ids = [f"query-{index:07d}" for index in range(len(hits))]
    started = time.perf_counter()
    report = build_report(frequencies, n_queries=len(hits), top_k=1, detail="summary")
    report_seconds = time.perf_counter() - started
    tracemalloc.start()
    summary_started = time.perf_counter()
    summary = probe_summary_to_json(
        ProbeResult(
            freqs=frequencies,
            hits_per_query=hits,
            query_ids=query_ids,
            report=report,
        )
    )
    summary_seconds = time.perf_counter() - summary_started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output_bytes = len(json.dumps(summary, separators=(",", ":")).encode("utf-8"))
    return {
        "corpus_size": size,
        "build_report_seconds": round(report_seconds, 4),
        "summary_seconds": round(summary_seconds, 4),
        "summary_bytes": output_bytes,
        "peak_bytes": peak,
        "lorenz_points_exported": summary["report"]["lorenz_points_exported"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--assert-targets", action="store_true")
    args = parser.parse_args()
    result = run(args.size)
    print(json.dumps(result, indent=2))
    if args.assert_targets:
        if result["build_report_seconds"] >= 3.0:
            raise SystemExit("build_report target missed: expected < 3 seconds")
        if result["summary_bytes"] >= 1_000_000:
            raise SystemExit("summary target missed: expected < 1 MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
