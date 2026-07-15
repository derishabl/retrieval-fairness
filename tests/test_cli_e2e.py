from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from retrieval_fairness.cli import main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(
        corpus,
        [
            {"id": "A", "text": "alpha document", "vector": [1.0, 0.0]},
            {"id": "B", "text": "beta document", "vector": [0.0, 1.0]},
            {"id": "C", "text": "gamma document", "vector": [-1.0, 0.0]},
        ],
    )
    _write_jsonl(
        queries,
        [
            {"id": "q1", "text": "alpha question", "vector": [1.0, 0.0]},
            {"id": "q2", "text": "beta question", "vector": [0.0, 1.0]},
        ],
    )
    return corpus, queries


def test_documented_cli_commands_end_to_end(tmp_path, capsys):
    corpus, queries = _inputs(tmp_path)
    baseline = tmp_path / "baseline.json"
    summary = tmp_path / "summary.json"
    dashboard = tmp_path / "probe.html"

    assert (
        main(
            [
                "probe",
                "--corpus",
                str(corpus),
                "--queries",
                str(queries),
                "--top-k",
                "1",
                "--json",
                str(baseline),
                "--summary-json",
                str(summary),
                "--html",
                str(dashboard),
                "--embedder-name",
                "fixture",
                "--embedder-revision",
                "r1",
                "--run-id",
                "test-run",
                "--git-commit",
                "deadbeef",
            ]
        )
        == 0
    )
    assert baseline.exists() and summary.exists() and dashboard.exists()

    diff_json = tmp_path / "diff.json"
    assert (
        main(
            [
                "diff",
                "--baseline",
                str(baseline),
                "--candidate",
                str(baseline),
                "--json",
                str(diff_json),
            ]
        )
        == 0
    )
    assert json.loads(diff_json.read_text(encoding="utf-8"))["mean_query_overlap"] == 1.0

    assert (
        main(
            [
                "gate",
                "--baseline",
                str(baseline),
                "--candidate",
                str(baseline),
                "--strict",
                "--max-coverage-drop",
                "0",
                "--min-query-overlap",
                "1",
            ]
        )
        == 0
    )

    rebuilt_dashboard = tmp_path / "dashboard.html"
    assert (
        main(
            [
                "dashboard",
                "--baseline",
                str(baseline),
                "--corpus",
                str(corpus),
                "--html",
                str(rebuilt_dashboard),
            ]
        )
        == 0
    )
    assert rebuilt_dashboard.exists()

    qrels = tmp_path / "qrels.json"
    qrels.write_text(json.dumps({"q1": {"A": 1}, "q2": {"B": 1}}), encoding="utf-8")
    qrels_out = tmp_path / "qrels-out.json"
    assert (
        main(
            [
                "qrels",
                "--probe",
                str(baseline),
                "--qrels",
                str(qrels),
                "--json",
                str(qrels_out),
            ]
        )
        == 0
    )
    assert json.loads(qrels_out.read_text(encoding="utf-8"))["micro_recall_at_k"] == 1.0

    synth_out = tmp_path / "synth.json"
    synth_summary = tmp_path / "synth-summary.json"
    assert (
        main(
            [
                "synth",
                "--corpus",
                str(corpus),
                "--top-k",
                "1",
                "--json",
                str(synth_out),
                "--summary-json",
                str(synth_summary),
            ]
        )
        == 0
    )
    assert synth_out.exists() and synth_summary.exists()
    assert "Traceback" not in capsys.readouterr().err


def test_probe_summary_only_gzip_and_error_paths(tmp_path, capsys):
    corpus, queries = _inputs(tmp_path)
    summary = tmp_path / "only-summary.json"
    assert (
        main(
            [
                "probe",
                "--corpus",
                str(corpus),
                "--queries",
                str(queries),
                "--summary-json",
                str(summary),
            ]
        )
        == 0
    )
    assert json.loads(summary.read_text(encoding="utf-8"))["summary_only"] is True

    compressed = tmp_path / "baseline.json"
    assert (
        main(
            [
                "probe",
                "--corpus",
                str(corpus),
                "--queries",
                str(queries),
                "--json",
                str(compressed),
                "--compress",
            ]
        )
        == 0
    )
    assert (tmp_path / "baseline.json.gz").exists()

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("not-json\n", encoding="utf-8")
    assert main(["probe", "--corpus", str(corpus), "--queries", str(invalid)]) == 2
    assert main(["probe", "--corpus", str(corpus), "--queries", str(queries), "--json", str(tmp_path)]) == 2
    assert "ОШИБКА" in capsys.readouterr().err


def test_gate_cli_strict_advisory_and_input_error(tmp_path, capsys):
    corpus, queries = _inputs(tmp_path)
    baseline = tmp_path / "wide.json"
    candidate = tmp_path / "narrow.json"
    common = ["--corpus", str(corpus), "--queries", str(queries), "--json"]
    assert main(["probe", *common, str(baseline), "--top-k", "3"]) == 0
    assert main(["probe", *common, str(candidate), "--top-k", "1"]) == 0
    gate = [
        "gate",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--max-coverage-drop",
        "0",
    ]
    assert main(gate) == 0
    assert main([*gate, "--strict"]) == 1
    assert main([*gate[:-2], "--max-coverage-drop", "2"]) == 2
    output = capsys.readouterr()
    assert "GATE FAILED" in output.out
    assert "ОШИБКА" in output.err


def test_demo_cli_entrypoints(capsys):
    assert main(["demo", "--top-k", "1"]) == 0
    assert main(["demo-diff", "--top-k", "1"]) == 0
    assert "Traceback" not in capsys.readouterr().err


def test_module_entrypoint_help_subprocess():
    completed = subprocess.run(
        [sys.executable, "-m", "retrieval_fairness", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0
    assert "probe" in completed.stdout
