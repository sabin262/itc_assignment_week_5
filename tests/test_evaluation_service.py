"""Unit tests for evaluation_service.py."""
from __future__ import annotations

import pytest

from app import evaluation_service
from app.schemas import RAGEvalResult


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _make_fake_llm():
    """Minimal stand-in that satisfies the langchain LLM interface."""
    return object()


def _make_fake_embeddings():
    return object()


def _fake_evaluate(*, scores: dict):
    """Return a callable that patches ragas.evaluate with fixed scores."""
    import pandas as pd

    class FakeResult:
        def to_pandas(self):
            return pd.DataFrame([scores])

    return lambda **_kwargs: FakeResult()


# ---------------------------------------------------------------------------
# evaluate_rag_response — happy path
# ---------------------------------------------------------------------------

def test_returns_rag_eval_result_with_scores(monkeypatch):
    scores = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.85,
        "context_precision": 0.75,
        "context_recall": 0.8,
    }

    monkeypatch.setattr(
        evaluation_service,
        "_run_ragas",
        _fake_evaluate(scores=scores),
        raising=False,
    )

    # Patch the heavy imports inside evaluate_rag_response
    import types
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _FakeDataset
    fake_ragas = types.ModuleType("ragas")
    fake_ragas.evaluate = _fake_evaluate(scores=scores)
    fake_metrics = types.ModuleType("ragas.metrics")
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        setattr(fake_metrics, name, name)

    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets)
    monkeypatch.setitem(__import__("sys").modules, "ragas", fake_ragas)
    monkeypatch.setitem(__import__("sys").modules, "ragas.metrics", fake_metrics)

    result = evaluation_service.evaluate_rag_response(
        question="What is the rent?",
        answer="The rent is £1,500 per month.",
        contexts=["Clause 4: monthly rent is £1,500."],
        langchain_llm=_make_fake_llm(),
        langchain_embeddings=_make_fake_embeddings(),
    )

    assert isinstance(result, RAGEvalResult)
    assert result.faithfulness == pytest.approx(0.9)
    assert result.answer_relevancy == pytest.approx(0.85)
    assert result.context_precision == pytest.approx(0.75)
    assert result.context_recall == pytest.approx(0.8)


class _FakeDataset:
    @staticmethod
    def from_dict(data):
        return _FakeDataset()


# ---------------------------------------------------------------------------
# evaluate_rag_response — empty contexts
# ---------------------------------------------------------------------------

def test_returns_none_scores_when_no_contexts():
    result = evaluation_service.evaluate_rag_response(
        question="What is the rent?",
        answer="The rent is £1,500 per month.",
        contexts=[],
        langchain_llm=_make_fake_llm(),
        langchain_embeddings=_make_fake_embeddings(),
    )

    assert isinstance(result, RAGEvalResult)
    assert result.faithfulness is None
    assert result.answer_relevancy is None
    assert result.context_precision is None
    assert result.context_recall is None


# ---------------------------------------------------------------------------
# evaluate_rag_response — missing ragas package
# ---------------------------------------------------------------------------

def test_raises_evaluation_error_when_ragas_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.setitem(sys.modules, "ragas", None)

    with pytest.raises(evaluation_service.EvaluationError, match="not installed"):
        evaluation_service.evaluate_rag_response(
            question="q",
            answer="a",
            contexts=["some context"],
            langchain_llm=_make_fake_llm(),
            langchain_embeddings=_make_fake_embeddings(),
        )


# ---------------------------------------------------------------------------
# evaluate_rag_response — NaN scores handled gracefully
# ---------------------------------------------------------------------------

def test_nan_scores_returned_as_none(monkeypatch):
    import math
    import sys
    import types

    nan = float("nan")
    scores = {
        "faithfulness": nan,
        "answer_relevancy": 0.7,
        "context_precision": nan,
        "context_recall": 0.6,
    }

    import pandas as pd

    class FakeResult:
        def to_pandas(self):
            return pd.DataFrame([scores])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _FakeDataset
    fake_ragas = types.ModuleType("ragas")
    fake_ragas.evaluate = lambda **_: FakeResult()
    fake_metrics = types.ModuleType("ragas.metrics")
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        setattr(fake_metrics, name, name)

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)

    result = evaluation_service.evaluate_rag_response(
        question="q",
        answer="a",
        contexts=["ctx"],
        langchain_llm=_make_fake_llm(),
        langchain_embeddings=_make_fake_embeddings(),
    )

    assert result.faithfulness is None
    assert result.answer_relevancy == pytest.approx(0.7)
    assert result.context_precision is None
    assert result.context_recall == pytest.approx(0.6)
