"""
Advanced RAG Retrieval Pipeline
=================================
Wraps ExperimentMemory with structured context building for agents.

Supports:
- Retrieval-augmented model selection
- Retrieval-augmented preprocessing decisions
- Conversational experiment querying
- Query classification for smart routing
"""

from __future__ import annotations

import re
from typing import Any

from app.memory.experiment_memory import ExperimentMemory
from app.utils.logger import get_logger

logger = get_logger("rag_retrieval")


class RAGRetriever:
    """
    Advanced RAG pipeline over experiment memory.

    Provides structured context augmentation for downstream agents
    by retrieving semantically similar past experiments and building
    LLM-ready context blocks.
    """

    def __init__(self, n_results: int = 5) -> None:
        self.n_results = n_results
        try:
            self.memory = ExperimentMemory()
        except Exception as e:
            logger.warning("memory_init_failed", error=str(e))
            self.memory = None  # type: ignore[assignment]

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve_for_model_selection(self, dataset_description: str, task_type: str) -> str:
        """
        Retrieve past experiments to augment model selection decisions.

        Returns a formatted context block ready for LLM injection.
        """
        query = f"{task_type} dataset: {dataset_description}"
        similar = self._safe_retrieve(query)
        if not similar:
            return "No similar past experiments found for model selection guidance."

        lines = ["Past experiments for model selection guidance:"]
        for i, exp in enumerate(similar, 1):
            meta = exp.get("metadata", {})
            lines.append(
                f"{i}. [{meta.get('task_type', '?')}] Best model: {meta.get('best_model', '?')} "
                f"| Metrics: {meta.get('best_metrics', '{}')} "
                f"| Dataset size: {meta.get('n_rows', '?')} rows"
            )
        return "\n".join(lines)

    def retrieve_for_preprocessing(self, dataset_description: str) -> str:
        """
        Retrieve past preprocessing decisions from similar experiments.
        Helps DataCleaningAgent and FeatureEngineeringAgent make better choices.
        """
        query = f"preprocessing cleaning features: {dataset_description}"
        similar = self._safe_retrieve(query)
        if not similar:
            return "No similar preprocessing history available."

        lines = ["Past preprocessing decisions for similar datasets:"]
        for i, exp in enumerate(similar, 1):
            doc = exp.get("document", "")
            meta = exp.get("metadata", {})
            lines.append(f"{i}. [{meta.get('filename', '?')}] {doc[:200]}...")
        return "\n".join(lines)

    def retrieve_for_chat(self, user_question: str) -> str:
        """
        Retrieve experiment context to answer a user question.
        Used by the conversational assistant.
        """
        similar = self._safe_retrieve(user_question)
        if not similar:
            return "No relevant past experiments found."

        lines = ["Relevant experiment history:"]
        for i, exp in enumerate(similar, 1):
            meta = exp.get("metadata", {})
            doc = exp.get("document", "")
            lines.append(
                f"\n--- Experiment {i}: {meta.get('experiment_id', '?')} ---\n"
                f"Dataset: {meta.get('filename', '?')}, "
                f"Task: {meta.get('task_type', '?')}, "
                f"Best model: {meta.get('best_model', '?')}\n"
                f"Summary: {doc[:300]}"
            )
        return "\n".join(lines)

    def get_experiment_count(self) -> int:
        """Return the number of stored experiments."""
        if self.memory is None:
            return 0
        try:
            return self.memory.get_experiment_count()
        except Exception:
            return 0

    def get_all_experiments(self) -> list[dict[str, Any]]:
        """Return all stored experiments for the Experiments page."""
        if self.memory is None:
            return []
        try:
            return self.memory.get_all_experiments()
        except Exception as e:
            logger.warning("get_all_experiments_failed", error=str(e))
            return []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _safe_retrieve(self, query: str) -> list[dict[str, Any]]:
        """Safe wrapper around memory retrieval — never raises."""
        if self.memory is None:
            return []
        try:
            return self.memory.find_similar(query, n_results=self.n_results)
        except Exception as e:
            logger.warning("retrieval_failed", query=query[:80], error=str(e))
            return []


# ── Singleton convenience ─────────────────────────────────────────────────────

_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    """Return a cached RAGRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
