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

__version__ = "0.2.0"

from retrieval_fairness.adapters import InMemoryVectorStore
from retrieval_fairness.diff import DiffReport, diff_reports
from retrieval_fairness.metrics import (
    FairnessReport,
    coverage,
    dark_matter,
    gini,
    hub_capture,
    lorenz,
    reachability_ceiling,
)
from retrieval_fairness.probe import ProbeResult, probe
from retrieval_fairness.provenance import ProbeMetadata
from retrieval_fairness.qrels import QrelsValidation, validate_qrels
from retrieval_fairness.serialize import load_probe, save_probe
from retrieval_fairness.types import Chunk, Hit, Query, VectorStore

__all__ = [
    "Chunk",
    "DiffReport",
    "FairnessReport",
    "Hit",
    "InMemoryVectorStore",
    "ProbeMetadata",
    "ProbeResult",
    "QrelsValidation",
    "Query",
    "VectorStore",
    "__version__",
    "coverage",
    "dark_matter",
    "diff_reports",
    "gini",
    "hub_capture",
    "load_probe",
    "lorenz",
    "probe",
    "reachability_ceiling",
    "save_probe",
    "validate_qrels",
]
