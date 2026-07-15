"""
embedders.py — контракт Embedder + реализации.

Embedder.encode(texts) -> matrix. Приводит любой эмбеддер (TF-IDF,
sentence-transformers, ...) к общему интерфейсу, чтобы probe/synth
работали поверх любого, не зная внутренностей.

Шаг 9.2: TfidfEmbedder (есть, лёгкий) + SentenceTransformerEmbedder
(реалистичный, optional dependency).
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Контракт эмбеддера: fit на корпусе, encode произвольных текстов."""

    def fit(self, texts: list[str]) -> Embedder: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


class TfidfEmbedder:
    """Лёгкий TF-IDF эмбеддер (sklearn). Без тяжёлых моделей.

    ВНИМАНИЕ про память: encode() денсифицирует sparse-матрицу под
    FAISS. Без max_features словарь на большом корпусе — сотни тысяч
    измерений, и dense-матрица не влезет в RAM (260k чанков × 400k
    слов ≈ сотни ГБ). Для корпусов >~10k чанков задавайте
    max_features (например 4096).
    """

    # порог предупреждения: оценка dense-матрицы в байтах (float64)
    _DENSE_WARN_BYTES = 2 * 1024**3  # 2 GB

    def __init__(self, ngram_range: tuple[int, int] = (1, 1), max_features: int | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self._fitted = False
        self._dim_val = 0

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        self._vec.fit(texts)
        self._fitted = True
        self._dim_val = len(self._vec.vocabulary_)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder not fitted; call .fit() first")
        est = len(texts) * self._dim_val * 8  # float64
        if est > self._DENSE_WARN_BYTES:
            warnings.warn(
                f"TfidfEmbedder.encode: dense-матрица ~{est / 1024**3:.1f} GB "
                f"({len(texts)} текстов × {self._dim_val} измерений). "
                "Риск OOM — задайте max_features (например 4096).",
                ResourceWarning,
                stacklevel=2,
            )
        return self._vec.transform(texts).toarray().astype(float)

    @property
    def dim(self) -> int:
        return self._dim_val


class SentenceTransformerEmbedder:
    """
    Dense-эмбеддер через sentence-transformers (all-MiniLM-L6-v2 по умолчанию).
    Optional: pip install 'retrieval-fairness[models]'.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers: "
                "pip install 'retrieval-fairness[models]'"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._dim_val = self._model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        # dense-эмбеддеры не требуют fit на корпусе; no-op
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(self._model.encode(texts, show_progress_bar=False), dtype=float)

    @property
    def dim(self) -> int:
        return self._dim_val


class FastembedEmbedder:
    """
    Dense-эмбеддер через fastembed (ONNX на CPU, без тяжёлого torch).
    Лёгкая альтернатива sentence-transformers; хорошо работает на CPU.
    Optional: pip install fastembed.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError("FastembedEmbedder requires fastembed: pip install fastembed") from e
        self._model = TextEmbedding(model_name=model_name)
        # dim узнаётся лениво при первом encode
        self._dim_val: int | None = None

    def fit(self, texts: list[str]) -> FastembedEmbedder:
        return self  # no-op для dense

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = list(self._model.embed(texts))
        arr = np.array(vecs, dtype=float)
        if self._dim_val is None and arr.size > 0:
            self._dim_val = arr.shape[1]
        return arr

    @property
    def dim(self) -> int:
        if self._dim_val is None:
            raise RuntimeError("dim unknown until first encode()")
        return self._dim_val


def get_embedder(name: str, **kw) -> Embedder:
    """Реестр эмбеддеров: 'tfidf' | 'minilm' | 'sbert' | 'fastembed' | 'bge'.

    minilm — фиксированная модель all-MiniLM-L6-v2 (model_name игнорируется).
    sbert  — sentence-transformers с model_name из kw (None = дефолт ST).
    bge    — fastembed с фиксированной BAAI/bge-small-en-v1.5.
    fastembed — fastembed с model_name из kw (дефолт BGE-small).
    """
    name = name.lower()
    if name == "tfidf":
        return TfidfEmbedder(**kw)
    if name == "minilm":
        # фиксированная модель; model_name не передаём, чтобы не переопределять
        kw.pop("model_name", None)
        return SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if name == "sbert":
        model_name = kw.pop("model_name", None)
        if model_name is not None:
            return SentenceTransformerEmbedder(model_name=model_name)
        return SentenceTransformerEmbedder()  # дефолт ST
    if name == "bge":
        kw.pop("model_name", None)  # фиксированная модель
        return FastembedEmbedder(model_name="BAAI/bge-small-en-v1.5")
    if name == "fastembed":
        model_name = kw.pop("model_name", "BAAI/bge-small-en-v1.5")
        return FastembedEmbedder(model_name=model_name)
    raise ValueError(f"unknown embedder: {name}")
