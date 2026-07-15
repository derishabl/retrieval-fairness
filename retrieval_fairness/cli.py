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
import contextlib
import json
import sys

from retrieval_fairness.adapters import InMemoryVectorStore
from retrieval_fairness.probe import probe
from retrieval_fairness.types import Chunk, Query


def _cli_error(msg: str) -> int:
    """Печать человекочитаемой ошибки CLI без трейсбека + exit 2."""
    print(f"ОШИБКА: {msg}", file=sys.stderr)
    return 2


def _wrap_cli(fn):
    """Декоратор CLI-команды: ловит ошибки ввода -> exit 2 с человеком-сообщением.
    Покрывает: нет файла, нет обязательного аргумента, невалидный аргумент сторa."""
    import functools

    @functools.wraps(fn)
    def wrapper(args):
        try:
            return fn(args)
        except FileNotFoundError as e:
            return _cli_error(f"файл не найден: {e.filename or e}")
        except PermissionError as e:
            return _cli_error(f"нет доступа к файлу: {e.filename or e}")
        except (ValueError, KeyError) as e:
            return _cli_error(str(e))
        except OSError as e:
            return _cli_error(f"ошибка ввода-вывода: {e}")
        except ImportError as e:
            return _cli_error(f"отсутствует зависимость: {e}. Установите optional-dep, см. docs/adapters.md")

    return wrapper


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


def _validate_store_args(args: argparse.Namespace) -> None:
    """Validate store-specific arguments before importing an adapter."""
    store_name = getattr(args, "store", "inmemory")
    required = {
        "inmemory": (("corpus", "--corpus"),),
        "faiss": (("index_path", "--index-path"),),
        "pgvector": (("database_url", "--database-url"),),
        "qdrant": (("url", "--url"), ("collection", "--collection")),
    }
    for attribute, option in required.get(store_name, ()):
        if not getattr(args, attribute, None):
            raise ValueError(f"{option} обязателен для --store {store_name}")


def _build_store_from_args(args, corpus: list[Chunk] | None = None):
    """Построить стор по --store и connection-аргументам."""
    _validate_store_args(args)
    store_name = getattr(args, "store", "inmemory")
    if store_name == "inmemory":
        if corpus is None:
            raise ValueError("inmemory store requires --corpus")
        return InMemoryVectorStore(corpus)
    if store_name == "faiss":
        from retrieval_fairness.adapters.faiss import FaissAdapter

        return FaissAdapter(
            index_path=args.index_path,
            ids_map_path=getattr(args, "ids_map", None),
            allow_legacy_ids_map=getattr(args, "allow_legacy_faiss_ids_map", False),
        )
    if store_name == "pgvector":
        from retrieval_fairness.adapters.pgvector import PgvectorAdapter

        return PgvectorAdapter(
            database_url=args.database_url,
            table=getattr(args, "table", "docs"),
            column=getattr(args, "column", "embedding"),
        )
    if store_name == "qdrant":
        from retrieval_fairness.adapters.qdrant import QdrantAdapter

        return QdrantAdapter(url=args.url, collection=args.collection, api_key=getattr(args, "api_key", None))
    raise ValueError(f"unknown store: {store_name}")


def _add_store_args(p) -> None:
    """Добавить --store + connection-аргументы в subparser."""
    p.add_argument("--store", choices=["inmemory", "faiss", "pgvector", "qdrant"], default="inmemory")
    # inmemory: --corpus (уже есть отдельно)
    # faiss
    p.add_argument("--index-path", help="FAISS: путь к .faiss индексу")
    p.add_argument("--ids-map", help="FAISS: checksum-bound JSON manifest")
    p.add_argument(
        "--allow-legacy-faiss-ids-map",
        action="store_true",
        help="FAISS: explicitly accept an unbound legacy {ids:[...]} sidecar",
    )
    # pgvector
    p.add_argument("--database-url", help="pgvector: postgres connection string")
    p.add_argument("--table", default="docs", help="pgvector: таблица")
    p.add_argument("--column", default="embedding", help="pgvector: векторная колонка")
    # qdrant
    p.add_argument("--url", help="Qdrant: endpoint")
    p.add_argument("--collection", help="Qdrant: коллекция")
    p.add_argument("--api-key", help="Qdrant: API key (cloud)")


@_wrap_cli
def cmd_probe(args: argparse.Namespace) -> int:
    _validate_store_args(args)
    corpus = _load_corpus(args.corpus) if args.corpus else None
    queries = _load_queries(args.queries)
    store = _build_store_from_args(args, corpus=corpus)
    report_detail = "summary" if args.summary_json and not args.json and not args.html else "full"
    result = probe(
        store,
        queries,
        top_k=args.top_k,
        corpus_texts={chunk.id: chunk.text for chunk in corpus} if corpus else None,
        workload_revision=args.workload_revision,
        corpus_revision=args.corpus_revision,
        embedder=args.embedder_name,
        embedder_revision=args.embedder_revision,
        run_id=args.run_id,
        git_commit=args.git_commit,
        report_detail=report_detail,
    )
    if result.report is None:
        raise RuntimeError("probe returned no report")
    print(result.report)
    if args.json:
        from retrieval_fairness.serialize import save_probe

        save_probe(result, args.json, compress=args.compress)
        suffix = ".gz" if args.compress and not args.json.endswith(".gz") else ""
        print(f"\nJSON-отчёт (baseline) сохранён: {args.json}{suffix}")
    if args.summary_json:
        from retrieval_fairness.serialize import save_probe_summary

        save_probe_summary(
            result,
            args.summary_json,
            max_exported_dark_ids=args.max_exported_dark_ids,
            max_lorenz_points=args.max_lorenz_points,
        )
        print(f"\nSummary JSON сохранён: {args.summary_json}")
    if args.html:
        from retrieval_fairness.dashboard import render_dashboard

        vecs = [c.vector for c in corpus] if corpus else None
        ids = [c.id for c in corpus] if corpus else None
        render_dashboard(result, args.html, chunks_vectors=vecs, chunk_ids=ids)
        print(f"HTML-дашборд сохранён: {args.html}")
    return 0


@_wrap_cli
def cmd_demo(args: argparse.Namespace) -> int:
    from retrieval_fairness.demo import run_demo

    run_demo(top_k=args.top_k)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    from retrieval_fairness.diff import diff_reports
    from retrieval_fairness.serialize import load_probe

    try:
        base = load_probe(args.baseline)
        cand = load_probe(args.candidate)
        d = diff_reports(
            base,
            cand,
            corpus_policy=args.corpus_policy,
            workload_policy=args.workload_policy,
        )
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    print(d)
    if args.json:
        import json as _json

        with open(args.json, "w", encoding="utf-8") as f:
            _json.dump(d.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nJSON-diff сохранён: {args.json}")
    return 0


@_wrap_cli
def cmd_demo_diff(args: argparse.Namespace) -> int:
    from retrieval_fairness.demo import run_migration_diff_demo

    run_migration_diff_demo(top_k=args.top_k)
    return 0


@_wrap_cli
def cmd_dashboard(args: argparse.Namespace) -> int:
    from retrieval_fairness.dashboard import render_dashboard_from_baseline

    render_dashboard_from_baseline(args.baseline, args.html, corpus_path=args.corpus)
    print(f"HTML-дашборд сохранён: {args.html}")
    return 0


@_wrap_cli
def cmd_qrels(args: argparse.Namespace) -> int:
    from retrieval_fairness.qrels import validate_qrels

    res = validate_qrels(
        args.probe,
        args.qrels,
        args.queries,
        min_relevance_grade=args.min_relevance_grade,
    )
    print(res)
    if args.json:
        import json as _json

        with open(args.json, "w", encoding="utf-8") as f:
            _json.dump(res.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nJSON сохранён: {args.json}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from retrieval_fairness.gate import run_gate_cli

    return run_gate_cli(args)


@_wrap_cli
def cmd_synth(args: argparse.Namespace) -> int:
    from retrieval_fairness.synth import synth_probe

    corpus = _load_corpus(args.corpus)
    result = synth_probe(
        corpus, top_k=args.top_k, n_per_chunk=args.n_per_chunk, n_terms=args.n_terms, query_style=args.style
    )
    print(result.report)
    if args.json:
        from retrieval_fairness.serialize import save_probe

        save_probe(result, args.json, compress=args.compress)
        suffix = ".gz" if args.compress and not args.json.endswith(".gz") else ""
        print(f"\nJSON-отчёт (baseline) сохранён: {args.json}{suffix}")
    if args.summary_json:
        from retrieval_fairness.serialize import save_probe_summary

        save_probe_summary(
            result,
            args.summary_json,
            max_exported_dark_ids=args.max_exported_dark_ids,
            max_lorenz_points=args.max_lorenz_points,
        )
        print(f"\nSummary JSON сохранён: {args.summary_json}")
    if args.html:
        from retrieval_fairness.dashboard import render_dashboard

        render_dashboard(
            result, args.html, chunks_vectors=[c.vector for c in corpus], chunk_ids=[c.id for c in corpus]
        )
        print(f"HTML-дашборд сохранён: {args.html}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows-консоль часто в cp866/cp1251 — русский вывод превращается в кракозябры.
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="retrieval_fairness",
        description="exposure-аудит векторного поиска: coverage, dark matter, Gini, CI-гейт",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="прогнать workload и напечатать отчёт exposure")
    p_probe.add_argument("--corpus", help="JSONL: {id, vector, text?} (для inmemory / PCA в дашборде)")
    p_probe.add_argument("--queries", required=True, help="JSONL: {id, vector, text?}")
    p_probe.add_argument("--top-k", type=int, default=10)
    p_probe.add_argument("--json", help="путь для полного JSON-экспорта")
    p_probe.add_argument("--summary-json", help="компактный JSON без raw frequencies/hits")
    p_probe.add_argument("--max-exported-dark-ids", type=int, default=0)
    p_probe.add_argument("--max-lorenz-points", type=int, default=512)
    p_probe.add_argument("--workload-revision", help="revision for queries without source text")
    p_probe.add_argument("--corpus-revision", help="revision for stores without source chunk content")
    p_probe.add_argument("--embedder-name", help="embedder/model family recorded in provenance")
    p_probe.add_argument("--embedder-revision", help="immutable model revision recorded in provenance")
    p_probe.add_argument("--run-id", help="caller-provided reproducible run ID")
    p_probe.add_argument("--git-commit", help="caller-provided source commit")
    p_probe.add_argument("--compress", action="store_true", help="gzip для полного JSON")
    p_probe.add_argument("--html", help="путь для HTML-дашборда")
    _add_store_args(p_probe)
    p_probe.set_defaults(func=cmd_probe)

    p_demo = sub.add_parser("demo", help="демо на синтетическом корпусе")
    p_demo.add_argument("--top-k", type=int, default=5)
    p_demo.set_defaults(func=cmd_demo)

    p_diff = sub.add_parser("diff", help="сравнить два baseline JSON")
    p_diff.add_argument("--baseline", required=True)
    p_diff.add_argument("--candidate", required=True)
    p_diff.add_argument("--json", help="путь для JSON-экспорта diff")
    p_diff.add_argument(
        "--corpus-policy",
        choices=["same-content", "same-ids", "allow-change", "same"],
        default="same-content",
    )
    p_diff.add_argument("--workload-policy", choices=["same-content", "same-ids"], default="same-content")
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
    p_gate.add_argument(
        "--max-coverage-drop",
        type=float,
        default=None,
        help="макс. падение coverage: доля 0..1 (0.05 = 5%%); 0 = zero tolerance",
    )
    p_gate.add_argument(
        "--max-dark-matter-rise",
        type=float,
        default=None,
        help="макс. рост dark-matter: доля 0..1 (0.05 = 5%%); 0 = zero tolerance",
    )
    p_gate.add_argument(
        "--max-gini-rise", type=float, default=None, help="макс. рост Gini (0..1; 0 = zero tolerance)"
    )
    p_gate.add_argument(
        "--min-query-overlap",
        type=float,
        default=None,
        help="мин. средний per-query overlap: доля 0..1 (0.8 = 80%%)",
    )
    p_gate.add_argument("--strict", action="store_true", help="нарушение -> exit 1 (для CI)")
    p_gate.add_argument(
        "--corpus-policy",
        choices=["same-content", "same-ids", "allow-change", "same"],
        default="same-content",
    )
    p_gate.add_argument("--workload-policy", choices=["same-content", "same-ids"], default="same-content")
    p_gate.add_argument(
        "--allow-legacy-alignment",
        action="store_true",
        help="разрешить небезопасное позиционное overlap-сравнение legacy baseline",
    )
    p_gate.set_defaults(func=cmd_gate)

    p_qrels = sub.add_parser("qrels", help="сверить dark matter с qrels: «потерянное золото» + recall@k")
    p_qrels.add_argument("--probe", required=True, help="save_probe JSON (probe --json / case_run --out)")
    p_qrels.add_argument("--qrels", required=True, help="qrels.json: {query_id: {doc_id: grade}}")
    p_qrels.add_argument("--queries", help="queries.jsonl (обязательно только для legacy schema v1)")
    p_qrels.add_argument(
        "--min-relevance-grade",
        type=int,
        default=1,
        help="relevant iff grade >= value (default: 1)",
    )
    p_qrels.add_argument("--json", help="экспорт результата в JSON")
    p_qrels.set_defaults(func=cmd_qrels)

    p_synth = sub.add_parser(
        "synth", help="antihub self-query аудит: синтетические запросы из корпуса (без query-логов)"
    )
    p_synth.add_argument("--corpus", required=True, help="JSONL: {id, text, vector}")
    p_synth.add_argument("--top-k", type=int, default=10)
    p_synth.add_argument("--n-per-chunk", type=int, default=1)
    p_synth.add_argument("--n-terms", type=int, default=5)
    p_synth.add_argument("--style", choices=["keywords", "text"], default="keywords")
    p_synth.add_argument("--json", help="путь для полного JSON-экспорта")
    p_synth.add_argument("--summary-json", help="компактный JSON")
    p_synth.add_argument("--max-exported-dark-ids", type=int, default=0)
    p_synth.add_argument("--max-lorenz-points", type=int, default=512)
    p_synth.add_argument("--compress", action="store_true", help="gzip для полного JSON")
    p_synth.add_argument("--html", help="путь для HTML-дашборда")
    p_synth.set_defaults(func=cmd_synth)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
