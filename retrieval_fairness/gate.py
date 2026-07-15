"""
gate.py — CI-гейт для retrieval-fairness.

Сравнивает candidate (новый прогон) с baseline по настраиваемым правилам.
Возвращает exit code: 0 = гейт пройден, 1 = нарушено правило (для CI).

Правила (по умолчанию advisory, opt-in strict). Пороги — доли 0..1
(0.05 = 5%); правило активно, если флаг передан; 0 = zero tolerance
(любое ухудшение = fail):
  --max-coverage-drop 0.05      coverage не должен упасть более чем на 5%
  --max-dark-matter-rise 0.05   dark-matter не должен вырасти более чем на 5%
  --max-gini-rise 0.1           Gini не должен вырасти более чем на 0.1
  --min-query-overlap 0.8       средний per-query overlap не ниже 0.8 (80%)
  --strict                      нарушения = exit 1 (иначе advisory, exit 0)

Использование в CI:
  python -m retrieval_fairness probe --corpus c.jsonl --queries q.jsonl --json new.json
  python -m retrieval_fairness gate --baseline v1.json --candidate new.json --strict \
      --max-coverage-drop 0.05 --max-dark-matter-rise 0.05
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from retrieval_fairness.diff import diff_reports
from retrieval_fairness.serialize import load_probe

# Все пороги — доли 0..1. Валидация защищает от молчаливого «5 = 500%»
# (доверие к CI-гейту: out-of-range порог = гейт, который никогда не сработает).
_PCT_RULES = {"coverage_drop", "dark_matter_rise", "query_overlap"}  # 0..1, показываем в %
_GINI_RULES = {"gini_rise"}  # 0..1, показываем как есть


def _validate_threshold(name: str, value: float | None, lo: float = 0.0, hi: float = 1.0) -> None:
    """Проверить, что порог в допустимом диапазоне; иначе ValueError."""
    if value is None:
        return
    if not (lo <= value <= hi):
        raise ValueError(
            f"--{name.replace('_', '-')}={value} вне диапазона [{lo}, {hi}] "
            f"(доли 0..1; 0.05 = 5%). Используйте долю, не проценты."
        )


def _fmt(name: str, value: float) -> str:
    """Отформатировать значение с единицами по типу правила."""
    if name in _PCT_RULES:
        return f"{value * 100:.2f}%"
    if name in _GINI_RULES:
        return f"{value:+.4f}"
    return f"{value:.4f}"


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
            cmp = "<=" if r.direction in ("drop", "rise") else ">="
            lines.append(
                f"  [{mark}] {r.name:22} actual={_fmt(r.name, r.actual):>10}  "
                f"{cmp} threshold={_fmt(r.name, r.threshold):>10}"
            )
        verdict = "GATE PASSED" if self.passed else "GATE FAILED"
        lines.append("-" * 60)
        lines.append(f"  {verdict}")
        lines.append("=" * 60)
        return "\n".join(lines)


def evaluate_gate(
    baseline_path: str,
    candidate_path: str,
    max_coverage_drop: float | None = None,  # доли (0..1); None = правило выключено
    max_dark_matter_rise: float | None = None,
    max_gini_rise: float | None = None,
    min_query_overlap: float | None = None,
    corpus_policy: str = "same-content",
    workload_policy: str = "same-content",
    allow_legacy_alignment: bool = False,
) -> GateResult:
    """
    Оценить правила гейта. Правило активно, если порог задан (не None);
    0 означает zero tolerance (любое ухудшение = fail).
    """
    # Валидация порогов ДО загрузки/диффа — out-of-range = явная ошибка,
    # не молчаливый pass (и не тратим время на загрузку при невалидном пороге).
    _validate_threshold("max_coverage_drop", max_coverage_drop)
    _validate_threshold("max_dark_matter_rise", max_dark_matter_rise)
    _validate_threshold("max_gini_rise", max_gini_rise)
    _validate_threshold("min_query_overlap", min_query_overlap)

    base = load_probe(baseline_path, strict_integrity=True)
    cand = load_probe(candidate_path, strict_integrity=True)
    if (
        min_query_overlap is not None
        and (not base.query_ids or not cand.query_ids)
        and not allow_legacy_alignment
    ):
        raise ValueError(
            "overlap gate requires query IDs; pass --allow-legacy-alignment "
            "to opt into unsafe positional alignment"
        )
    d = diff_reports(
        base,
        cand,
        corpus_policy=corpus_policy,
        workload_policy=workload_policy,
    )

    rules: list[GateRule] = []

    if max_coverage_drop is not None:
        # coverage_delta = c - b; падение = -delta. Падение > max => fail
        drop = -d.coverage_delta
        passed = drop <= max_coverage_drop
        rules.append(GateRule("coverage_drop", drop, max_coverage_drop, passed, "drop"))

    if max_dark_matter_rise is not None:
        rise = d.dark_matter_delta
        passed = rise <= max_dark_matter_rise
        rules.append(GateRule("dark_matter_rise", rise, max_dark_matter_rise, passed, "rise"))

    if max_gini_rise is not None:
        rise = d.gini_delta
        passed = rise <= max_gini_rise
        rules.append(GateRule("gini_rise", rise, max_gini_rise, passed, "rise"))

    if min_query_overlap is not None:
        ov = d.mean_query_overlap
        passed = ov >= min_query_overlap
        rules.append(GateRule("query_overlap", ov, min_query_overlap, passed, "min"))

    all_passed = all(r.passed for r in rules) if rules else True
    return GateResult(passed=all_passed, rules=rules)


def run_gate_cli(args) -> int:
    """CLI handler. Возвращает exit code.
    0 = гейт пройден; 1 = нарушено правило (strict); 2 = ошибка ввода
    (невалидный порог / файл) — печатаем человекочитаемо, без трейсбека."""
    try:
        res = evaluate_gate(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            max_coverage_drop=args.max_coverage_drop,
            max_dark_matter_rise=args.max_dark_matter_rise,
            max_gini_rise=args.max_gini_rise,
            min_query_overlap=args.min_query_overlap,
            corpus_policy=getattr(args, "corpus_policy", "same-content"),
            workload_policy=getattr(args, "workload_policy", "same-content"),
            allow_legacy_alignment=(
                getattr(args, "allow_legacy_alignment", False) or not getattr(args, "strict", False)
            ),
        )
    except ValueError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        print("Пороги gate — доли 0..1 (0.05 = 5%). 0 = zero tolerance.", file=sys.stderr)
        return 2
    except (FileNotFoundError, KeyError) as e:
        print(f"ОШИБКА: не удалось загрузить baseline/candidate: {e}", file=sys.stderr)
        print("Файл должен быть сохранён через `probe --json` (формат save_probe).", file=sys.stderr)
        return 2
    print(res)
    if not res.passed:
        if args.strict:
            print("\nGATE FAILED — strict mode, returning exit 1 (CI должен блокировать)")
            return 1
        else:
            print("\nGATE FAILED — advisory mode (без --strict), exit 0")
            return 0
    return 0
