"""
evaluate.py — оценочный стенд защиты PoisonGuard.

Меряет, насколько хорошо фильтр отличает чистые документы от нарушителей,
и показывает ПОКРЫТИЕ ПО КАТЕГОРИЯМ — где защита слепа и что усиливать.

Запуск:  python evaluate.py
"""

from __future__ import annotations
from collections import defaultdict

from poison_guard import PoisonGuard
from fixtures import CLEAN, all_samples


def evaluate() -> None:
    guard = PoisonGuard().fit(CLEAN)
    samples = all_samples()

    tp = fp = tn = fn = 0
    by_category: dict[str, list[bool]] = defaultdict(list)  # категория -> [пойман?]

    print("=" * 70)
    print("ДЕТАЛИ ПО ДОКУМЕНТАМ")
    print("=" * 70)
    for s in samples:
        res = guard.check(s.text)
        flagged = res.is_suspicious

        if s.is_poison and flagged:
            tp += 1
        elif s.is_poison and not flagged:
            fn += 1
        elif not s.is_poison and flagged:
            fp += 1
        else:
            tn += 1

        if s.is_poison:
            by_category[s.category].append(flagged)

        mark = "OK " if (flagged == s.is_poison) else "!! "
        kind = "нарушитель" if s.is_poison else "чистый    "
        print(f"  {mark}[{kind}] {s.category:22} score={res.score:.2f}  {s.text[:44]}...")

    # --- метрики ---
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("\n" + "=" * 70)
    print("МЕТРИКИ")
    print("=" * 70)
    print(f"  Поймано нарушителей (TP):        {tp}")
    print(f"  Пропущено нарушителей (FN):      {fn}   <- пробелы для усиления")
    print(f"  Ложных срабатываний (FP):        {fp}   <- бьёт по чистым")
    print(f"  Чистые прошли (TN):              {tn}")
    print(f"  ---")
    print(f"  Precision:  {precision:.2f}")
    print(f"  Recall:     {recall:.2f}   (доля пойманных нарушителей)")
    print(f"  F1:         {f1:.2f}")
    print(f"  FP-rate:    {fpr:.2f}   (доля ложных тревог на чистых)")

    print("\n" + "=" * 70)
    print("ПОКРЫТИЕ ПО КАТЕГОРИЯМ  (где защита слепа)")
    print("=" * 70)
    for cat, hits in sorted(by_category.items()):
        caught = sum(hits)
        total = len(hits)
        status = "покрыто" if caught == total else "ПРОБЕЛ"
        print(f"  {cat:24} {caught}/{total}  {status}")


if __name__ == "__main__":
    evaluate()
