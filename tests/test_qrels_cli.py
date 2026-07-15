"""test_qrels_cli.py — `retrieval-fairness qrels` CLI subcommand."""

from __future__ import annotations

import json
import os
import tempfile

from retrieval_fairness.cli import main as cli_main


def test_qrels_cli_runs():
    with tempfile.TemporaryDirectory() as d:
        probe = {"freqs": {"A": 2, "B": 1, "C": 0}, "hits_per_query": [["A", "B"], ["A"]], "report": {}}
        qrels = {"q1": {"C": 1}, "q2": {"A": 1}}
        pp = os.path.join(d, "p.json")
        with open(pp, "w", encoding="utf-8") as file:
            json.dump(probe, file)
        qp = os.path.join(d, "q.json")
        with open(qp, "w", encoding="utf-8") as file:
            json.dump(qrels, file)
        qq = os.path.join(d, "qql.jsonl")
        with open(qq, "w", encoding="utf-8") as f:
            f.write('{"id": "q1"}\n{"id": "q2"}\n')
        out = os.path.join(d, "out.json")
        rc = cli_main(["qrels", "--probe", pp, "--qrels", qp, "--queries", qq, "--json", out])
        assert rc == 0
        with open(out, encoding="utf-8") as file:
            res = json.load(file)
        assert res["dark_and_relevant"] == 1
        assert res["dark_relevant_ids"] == ["C"]
        assert res["recall_at_k"] == 0.5


def test_qrels_cli_missing_file_error_exit2():
    rc = cli_main(["qrels", "--probe", "nonexistent.json", "--qrels", "x.json", "--queries", "y.jsonl"])
    assert rc == 2  # _wrap_cli: FileNotFoundError -> exit 2
