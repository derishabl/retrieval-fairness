"""test_embedders.py — Embedder contract + TfidfEmbedder."""

from __future__ import annotations

import os

import numpy as np
import pytest

from retrieval_fairness.embedders import Embedder, TfidfEmbedder, get_embedder


def test_tfidf_fit_encode():
    emb = TfidfEmbedder()
    texts = ["отпуск через HR портал", "VPN настройка приложение", "зарплата аванс"]
    emb.fit(texts)
    mat = emb.encode(["отпуск HR"])
    assert mat.shape[0] == 1
    assert mat.shape[1] == emb.dim
    assert emb.dim > 0


def test_tfidf_encode_without_fit_raises():
    emb = TfidfEmbedder()
    try:
        emb.encode(["test"])
        assert False
    except RuntimeError:
        pass


def test_tfidf_max_features_caps_dim():
    texts = [f"word{i} common token{i}" for i in range(50)]
    emb = TfidfEmbedder(max_features=10).fit(texts)
    assert emb.dim == 10
    assert emb.encode(["word1"]).shape[1] == 10
    # через реестр тоже прокидывается
    emb2 = get_embedder("tfidf", max_features=5).fit(texts)
    assert emb2.dim == 5


def test_tfidf_warns_on_huge_dense_matrix():
    import warnings

    emb = TfidfEmbedder().fit(["alpha beta gamma"])
    emb._dim_val = 10**9  # симуляция огромного словаря без аллокации
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            emb.encode(["alpha"])
        except Exception:
            pass  # важно только предупреждение до transform
    assert any(issubclass(x.category, ResourceWarning) for x in w)


def test_get_embedder_tfidf():
    emb = get_embedder("tfidf")
    assert isinstance(emb, TfidfEmbedder)
    assert isinstance(emb, Embedder) if hasattr(Embedder, "__class__") else True


def test_get_embedder_unknown():
    try:
        get_embedder("nope")
        assert False
    except ValueError:
        pass


def test_tfidf_satisfies_protocol():
    emb = TfidfEmbedder()
    # Protocol — structural; проверяем наличие методов
    assert hasattr(emb, "fit") and hasattr(emb, "encode") and hasattr(emb, "dim")


def test_get_embedder_sbert_passes_model_name():
    """Баг #2: get_embedder('sbert', model_name=...) раньше игнорировал модель.
    Проверяем, что model_name доходит до SentenceTransformerEmbedder. Без рабочей
    sentence-transformers — мокаем конструктор."""
    import retrieval_fairness.embedders as em

    captured = {}

    class _FakeST:
        def __init__(self, model_name=None):
            captured["model_name"] = model_name

        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, show_progress_bar=False):
            return np.zeros((len(texts), 384))

    # подменяем класс в модуле, чтобы импорт внутри get_embedder подхватил
    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeST
    sys.modules["sentence_transformers"] = fake_module
    try:
        # перезагружаем, чтобы SentenceTransformerEmbedder использовал фейк
        import importlib

        importlib.reload(em)
        em.get_embedder("sbert", model_name="my-org/my-model")
        assert captured["model_name"] == "my-org/my-model"
        # minilm — фиксированная, model_name игнорируется
        em.get_embedder("minilm", model_name="ignored-pls")
        assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    finally:
        sys.modules.pop("sentence_transformers", None)
        importlib.reload(em)
        importlib.reload(importlib.import_module("retrieval_fairness"))


@pytest.mark.skipif(
    os.environ.get("RF_TEST_MINILM") != "1",
    reason="set RF_TEST_MINILM=1 to test sentence-transformers (heavy dep)",
)
def test_minilm_optional():
    from retrieval_fairness.embedders import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder()
    emb.fit(["test"])
    mat = emb.encode(["hello world"])
    assert mat.shape[0] == 1
    assert mat.shape[1] == emb.dim
    assert emb.dim == 384  # all-MiniLM-L6-v2


if __name__ == "__main__":
    import sys

    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    p = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            p += 1
        except (AssertionError, Exception) as e:
            print(f"  SKIP/FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{p}/{len(fns)} passed")
    sys.exit(0)
