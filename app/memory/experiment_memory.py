"""
Experiment Memory
==================
High-level interface for storing and retrieving experiment memory.
Uses sentence-transformers for local embeddings + ChromaDB for storage.
"""

from __future__ import annotations

from typing import Any

from app.memory.chroma_store import ChromaStore
from app.memory.embeddings import EmbeddingModel
from app.utils.logger import get_logger

logger = get_logger("experiment_memory")


class ExperimentMemory:
    """Manages long-term experiment memory with semantic search."""

    def __init__(self) -> None:
        self.store = ChromaStore()
        self.embedder = EmbeddingModel()

    def store_experiment(
        self,
        experiment_id: str,
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store an experiment in ChromaDB with its embedding."""
        embedding = self.embedder.embed(document)
        self.store.add(
            doc_id=experiment_id,
            document=document,
            metadata=metadata,
            embedding=embedding,
        )
        logger.info("experiment_stored", exp_id=experiment_id)

    def find_similar(
        self, query: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Find semantically similar experiments."""
        if self.store.count() == 0:
            return []
        return self.store.query(query, n_results=n_results)

    def get_all_experiments(self) -> list[dict[str, Any]]:
        """Return all stored experiments."""
        return self.store.get_all()

    def get_experiment_count(self) -> int:
        return self.store.count()

    def rag_context(self, query: str, n_results: int = 3) -> str:
        """
        Build RAG context string from similar experiments.
        Useful for giving LLM agents context from past runs.
        """
        similar = self.find_similar(query, n_results=n_results)
        if not similar:
            return "No relevant past experiments found."

        lines = ["Relevant past experiments:"]
        for i, exp in enumerate(similar, 1):
            meta = exp.get("metadata", {})
            lines.append(
                f"{i}. [{meta.get('experiment_id', '?')}] "
                f"Dataset: {meta.get('filename', '?')}, "
                f"Task: {meta.get('task_type', '?')}, "
                f"Best model: {meta.get('best_model', '?')}"
            )
        return "\n".join(lines)
