"""
stores.py — совместимый shim (deprecated).

InMemoryVectorStore перенесён в retrieval_fairness.adapters.inmemory.
Этот модуль оставлен для обратной совместимости; новые импорты идут
из retrieval_fairness.adapters.
"""

from __future__ import annotations

from retrieval_fairness.adapters.inmemory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore"]
