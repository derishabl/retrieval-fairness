"""
cli.py — CLI retrieval_fairness.

Шаг 1: одна команда `probe` — прогнать workload по стору и напечатать
отчёт exposure. JSON-экспорт — для будущего regression-diff (Шаг 2).

Запуск:
  python -m retrieval_fairness probe --corpus corpus.jsonl --queries queries.jsonl --top-k 10
  python -m retrieval_fairness probe ... --json report.json
  python -m retrieval_fairness demo
"""

from __future__ import annotations
import argparse
import json
import sys

from retrieval_fairness.types import Chunk, Query
from retrieval_fairness.stores import InMemoryVectorStore
from retrieval_fairness.probe import probe


def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_corpus(path: str) -> list[Chunk]:
    rows = _load_jsonl(path)
    return [Chunk(id=r["id"], text=r.get("text", r["id"]), vector=r["vector"]) for r in rows]


def _load_queries(path: str) -> list[Query]:
    rows = _load_jsonl(path)
    return [Query(id=r["id"], vector=r["vector"], text=r.get("text", "")) for r in rows]


def cmd_probe(args: argparse.Namespace) -> int:
    corpus = _load_corpus(args.corpus)
    queries = _load_queries(args.queries)
    store = InMemoryVectorStore(corpus)
    result = probe(store, queries, top_k=args.top_k)
    assert result.report is not None
    print(result.report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nJSON-отчёт сохранён: {args.json}")
    if args.html:
        from retrieval_fairness.dashboard import render_dashboard
        render_dashboard(result, args.html,
                         chunks_vectors=[c.vector for c in corpus],
                         chunk_ids=[c.id for c in corpus])
        print(f"HTML-дашборд сохранён: {args.html}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from retrieval_fairness.demo import run_demo
    run_demo(top_k=args.top_k)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    from retrieval_fairness.serialize import load_probe
    from retrieval_fairness.diff import diff_reports
    base = load_probe(args.baseline)
    cand = load_probe(args.candidate)
    d = diff_reports(base, cand)
    print(d)
    if args.json:
        import json as _json
        with open(args.json, "w", encoding="utf-8") as f:
            _json.dump(d.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nJSON-diff сохранён: {args.json}")
    return 0


def cmd_demo_diff(args: argparse.Namespace) -> int:
    from retrieval_fairness.demo import run_migration_diff_demo
    run_migration_diff_demo(top_k=args.top_k)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from retrieval_fairness.dashboard import render_dashboard_from_baseline
    render_dashboard_from_baseline(args.baseline, args.html, corpus_path=args.corpus)
    print(f"HTML-дашборд сохранён: {args.html}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from retrieval_fairness.gate import run_gate_cli
    return run_gate_cli(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval_fairness", description="«code coverage для retrieval»")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="прогнать workload и напечатать отчёт exposure")
    p_probe.add_argument("--corpus", required=True, help="JSONL: {id, vector, text?}")
    p_probe.add_argument("--queries", required=True, help="JSONL: {id, vector, text?}")
    p_probe.add_argument("--top-k", type=int, default=10)
    p_probe.add_argument("--json", help="путь для JSON-экспорта отчёта")
    p_probe.add_argument("--html", help="путь для HTML-дашборда")
    p_probe.set_defaults(func=cmd_probe)

    p_demo = sub.add_parser("demo", help="демо на синтетическом корпусе")
    p_demo.add_argument("--top-k", type=int, default=5)
    p_demo.set_defaults(func=cmd_demo)

    p_diff = sub.add_parser("diff", help="сравнить два baseline JSON")
    p_diff.add_argument("--baseline", required=True)
    p_diff.add_argument("--candidate", required=True)
    p_diff.add_argument("--json", help="путь для JSON-экспорта diff")
    p_diff.set_defaults(func=cmd_diff)

    p_demo_diff = sub.add_parser("demo-diff", help="демо regression diff при смене эмбеддера")
    p_demo_diff.add_argument("--top-k", type=int, default=5)
    p_demo_diff.set_defaults(func=cmd_demo_diff)

    p_dash = sub.add_parser("dashboard", help="HTML-дашборд из baseline JSON")
    p_dash.add_argument("--baseline", required=True)
    p_dash.add_argument("--html", required=True)
    p_dash.add_argument("--corpus", help="JSONL с векторами для PCA-проекции")
    p_dash.set_defaults(func=cmd_dashboard)

    p_gate = sub.add_parser("gate", help="CI-гейт: сравнить candidate с baseline по правилам")
    p_gate.add_argument("--baseline", required=True)
    p_gate.add_argument("--candidate", required=True)
    p_gate.add_argument("--max-coverage-drop", type=float, default=0.0, help="макс. падение coverage (п.п., 0..1)")
    p_gate.add_argument("--max-dark-matter-rise", type=float, default=0.0)
    p_gate.add_argument("--max-gini-rise", type=float, default=0.0)
    p_gate.add_argument("--min-query-overlap", type=float, default=0.0)
    p_gate.add_argument("--strict", action="store_true", help="нарушение -> exit 1 (для CI)")
    p_gate.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
