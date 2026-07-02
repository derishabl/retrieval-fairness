"""
gate.py — CI-гейт для retrieval-fairness.

Сравнивает candidate (новый прогон) с baseline по настраиваемым правилам.
Возвращает exit code: 0 = гейт пройден, 1 = нарушено правило (для CI).

Правила (по умолчанию advisory, opt-in strict):
  --max-coverage-drop 5      coverage не должен упасть более чем на 5 п.п.
  --max-dark-matter-rise 5   dark-matter не должен вырасти более чем на 5 п.п.
  --max-gini-rise 0.1        Gini не должен вырасти более чем на 0.1
  --min-query-overlap 0.8    средний per-query overlap не ниже 0.8
  --strict                   нарушения = exit 1 (иначе advisory, exit 0)

Использование в CI:
  python -m retrieval_fairness probe --corpus c.jsonl --queries q.jsonl --json new.json
  python -m retrieval_fairness gate --baseline v1.json --candidate new.json --strict \
      --max-coverage-drop 5 --max-dark-matter-rise 5
"""

from __future__ import annotations
from dataclasses import dataclass, field
import sys

from retrieval_fairness.serialize import load_probe
from retrieval_fairness.diff import diff_reports


@dataclass
class GateRule:
    name: str
    actual: float
    threshold: float
    passed: bool
    direction: str  # "drop" (actual <= threshold ok) or "rise" / "min"


@dataclass
class GateResult:
    passed: bool
    rules: list[GateRule] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "GATE", "=" * 60]
        for r in self.rules:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.name:30} actual={r.actual:+.4f}  threshold={r.threshold:+.4f}")
        verdict = "GATE PASSED" if self.passed else "GATE FAILED"
        lines.append("-" * 60)
        lines.append(f"  {verdict}")
        lines.append("=" * 60)
        return "\n".join(lines)


def evaluate_gate(
    baseline_path: str,
    candidate_path: str,
    max_coverage_drop: float = 0.0,    # п.п. (0..1), 0 = правило выключено
    max_dark_matter_rise: float = 0.0,
    max_gini_rise: float = 0.0,
    min_query_overlap: float = 0.0,
) -> GateResult:
    """
    Оценить правила гейта. Правило активно, если порог > 0.
    """
    base = load_probe(baseline_path)
    cand = load_probe(candidate_path)
    d = diff_reports(base, cand)

    rules: list[GateRule] = []

    if max_coverage_drop > 0:
        # coverage_delta = c - b; падение = -delta. Падение > max => fail
        drop = -d.coverage_delta
        passed = drop <= max_coverage_drop
        rules.append(GateRule("coverage_drop", drop, max_coverage_drop, passed, "drop"))

    if max_dark_matter_rise > 0:
        rise = d.dark_matter_delta
        passed = rise <= max_dark_matter_rise
        rules.append(GateRule("dark_matter_rise", rise, max_dark_matter_rise, passed, "rise"))

    if max_gini_rise > 0:
        rise = d.gini_delta
        passed = rise <= max_gini_rise
        rules.append(GateRule("gini_rise", rise, max_gini_rise, passed, "rise"))

    if min_query_overlap > 0:
        ov = d.mean_query_overlap
        passed = ov >= min_query_overlap
        rules.append(GateRule("query_overlap", ov, min_query_overlap, passed, "min"))

    all_passed = all(r.passed for r in rules) if rules else True
    return GateResult(passed=all_passed, rules=rules)


def run_gate_cli(args) -> int:
    """CLI handler. Возвращает exit code."""
    res = evaluate_gate(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        max_coverage_drop=args.max_coverage_drop,
        max_dark_matter_rise=args.max_dark_matter_rise,
        max_gini_rise=args.max_gini_rise,
        min_query_overlap=args.min_query_overlap,
    )
    print(res)
    if not res.passed:
        if args.strict:
            print("\nGATE FAILED — strict mode, returning exit 1 (CI должен блокировать)")
            return 1
        else:
            print("\nGATE FAILED — advisory mode (без --strict), exit 0")
            return 0
    return 0
