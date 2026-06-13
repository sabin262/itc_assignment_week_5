"""RAGAS evaluation service for the RAG chat pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.schemas import RAGEvalResult


class EvaluationError(RuntimeError):
    """Raised when RAGAS evaluation cannot complete."""


def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: list[str],
    langchain_llm: Any,
    langchain_embeddings: Any,
) -> "RAGEvalResult":
    """
    Score a single RAG response using RAGAS metrics.

    Returns RAGEvalResult with faithfulness, answer_relevancy,
    context_precision, and context_recall scores (each 0.0–1.0).
    Returns None for a metric when RAGAS cannot compute it.
    """
    from app.schemas import RAGEvalResult

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise EvaluationError(
            "RAGAS or datasets package is not installed."
        ) from exc

    if not contexts:
        return RAGEvalResult(
            faithfulness=None,
            answer_relevancy=None,
            context_precision=None,
            context_recall=None,
        )

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        # ground_truth required by context_recall; use the answer itself as a
        # proxy when no human reference is available.
        "ground_truth": [answer],
    }

    try:
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=langchain_llm,
            embeddings=langchain_embeddings,
            raise_exceptions=False,
        )
        scores = result.to_pandas().iloc[0].to_dict()
    except Exception as exc:
        raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

    def _safe(key: str) -> float | None:
        val = scores.get(key)
        if val is None:
            return None
        try:
            f = float(val)
            return None if f != f else round(f, 4)  # NaN guard
        except (TypeError, ValueError):
            return None

    return RAGEvalResult(
        faithfulness=_safe("faithfulness"),
        answer_relevancy=_safe("answer_relevancy"),
        context_precision=_safe("context_precision"),
        context_recall=_safe("context_recall"),
    )
