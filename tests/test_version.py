"""test_version.py — гард: __version__ не расходится с pyproject.toml."""

from __future__ import annotations

import os
import re

import retrieval_fairness

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_version_matches_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
    assert m, "version не найдена в pyproject.toml"
    assert retrieval_fairness.__version__ == m.group(1), (
        f"__init__.__version__={retrieval_fairness.__version__} "
        f"!= pyproject={m.group(1)} — обновите оба при релизе"
    )
