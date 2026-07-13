"""
retrieval_fairness — exposure-аудит векторного поиска / RAG.

Показывает, какую долю векторного корпуса реально достают запросы,
а какую — никогда не находят (dark matter / antihub inventory); меряет
концентрацию exposure (Gini), захват хабами, и regression-diff при смене
эмбеддера. Почему dark matter неизбежен — см. hubness (Radovanović et al.,
JMLR 2010) и README «Why your index has dark matter».

Шаг 1: контракт VectorStore + InMemoryVectorStore + базовые метрики
(coverage, Gini, dark-matter, hub-capture) + CLI probe.
"""

__version__ = "0.1.1"

from retrieval_fairness.types import Chunk, Hit, Query, VectorStore
from retrieval_fairness.adapters import InMemoryVectorStore
from retrieval_fairness.metrics import (
    coverage,
    gini,
    reachability_ceiling,
    dark_matter,
    hub_capture,
    lorenz,
    FairnessReport,
)
from retrieval_fairness.probe import probe, ProbeResult
from retrieval_fairness.diff import diff_reports, DiffReport
from retrieval_fairness.serialize import save_probe, load_probe
from retrieval_fairness.qrels import validate_qrels, QrelsValidation

__all__ = [
    "__version__",
    "Chunk",
    "Hit",
    "Query",
    "VectorStore",
    "InMemoryVectorStore",
    "coverage",
    "gini",
    "reachability_ceiling",
    "dark_matter",
    "hub_capture",
    "lorenz",
    "FairnessReport",
    "probe",
    "ProbeResult",
    "diff_reports",
    "DiffReport",
    "save_probe",
    "load_probe",
    "validate_qrels",
    "QrelsValidation",
]
