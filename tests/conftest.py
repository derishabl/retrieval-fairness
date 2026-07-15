"""conftest.py — fallback для запуска тестов без `pip install -e .`.

При установленном пакете sys.path-хак не нужен, но безвреден.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
