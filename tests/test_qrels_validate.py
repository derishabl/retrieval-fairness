"""test_qrels_validate.py — smoke-тест scripts/qrels_validate.py на мини-фикстуре."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "qrels_validate.py"
)


def test_qrels_validate_smoke():
    with tempfile.TemporaryDirectory() as d:
        # корпус A,B,C,D; q1 нашёл A,B; q2 нашёл A. C,D — dark matter.
        # qrels: q1 релевантен C (потерянное золото), q2 релевантен A (найден).
        probe = {
            "freqs": {"A": 2, "B": 1, "C": 0, "D": 0},
            "hits_per_query": [["A", "B"], ["A"]],
            "report": {},
        }
        qrels = {"q1": {"C": 1}, "q2": {"A": 1}}
        pp, qp, qq, out = (
            os.path.join(d, n) for n in ("probe.json", "qrels.json", "queries.jsonl", "out.json")
        )
        with open(pp, "w", encoding="utf-8") as file:
            json.dump(probe, file)
        with open(qp, "w", encoding="utf-8") as file:
            json.dump(qrels, file)
        with open(qq, "w", encoding="utf-8") as f:
            f.write('{"id": "q1"}\n{"id": "q2"}\n')

        r = subprocess.run(
            [sys.executable, SCRIPT, "--probe", pp, "--qrels", qp, "--queries", qq, "--json", out],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        with open(out, encoding="utf-8") as file:
            res = json.load(file)
        assert res["dark_matter"] == 2
        assert res["relevant_in_corpus"] == 2  # A и C
        assert res["dark_and_relevant"] == 1  # C
        assert res["dark_relevant_ids"] == ["C"]
        assert res["qrels_pairs_total"] == 2
        assert res["qrels_pairs_in_topk"] == 1  # A найден, C нет
        assert res["recall_at_k"] == 0.5


def test_qrels_validate_mismatched_queries_errors():
    with tempfile.TemporaryDirectory() as d:
        probe = {"freqs": {"A": 1}, "hits_per_query": [["A"]], "report": {}}
        pp, qp, qq = (os.path.join(d, n) for n in ("p.json", "q.json", "qq.jsonl"))
        with open(pp, "w", encoding="utf-8") as file:
            json.dump(probe, file)
        with open(qp, "w", encoding="utf-8") as file:
            json.dump({}, file)
        with open(qq, "w", encoding="utf-8") as f:
            f.write('{"id": "q1"}\n{"id": "q2"}\n')  # 2 запроса vs 1 hits
        r = subprocess.run(
            [sys.executable, SCRIPT, "--probe", pp, "--qrels", qp, "--queries", qq],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 2
