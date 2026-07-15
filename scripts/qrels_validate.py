"""
qrels_validate.py — thin wrapper over retrieval_fairness.qrels.

The qrels "lost gold" cross-check (dark matter vs relevance) is now a
first-class package feature: see `retrieval_fairness.qrels` and the
`retrieval-fairness qrels` CLI command. This script stays as a
standalone entry point for the case study workflow.

Usage:
  python scripts/qrels_validate.py --probe cases/nq_bge_sample.json \
      --qrels data/nq/qrels.json --queries data/nq/queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_fairness.qrels import validate_qrels


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate dark matter against qrels (lost gold + recall@k)")
    ap.add_argument("--probe", required=True, help="save_probe JSON (probe --json / case_run --out)")
    ap.add_argument("--qrels", required=True, help="qrels.json: {query_id: {doc_id: grade}}")
    ap.add_argument("--queries", help="queries.jsonl (required only for legacy schema v1)")
    ap.add_argument("--min-relevance-grade", type=int, default=1)
    ap.add_argument("--json", help="export the validation result as JSON")
    args = ap.parse_args()

    try:
        res = validate_qrels(
            args.probe,
            args.qrels,
            args.queries,
            min_relevance_grade=args.min_relevance_grade,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(res)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"saved -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
