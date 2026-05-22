"""
Unit Tests — Memory & RAG System
====================================
Tests ChromaDB storage, embedding, similarity search, and RAG context building.
Uses real temp directories for ChromaDB (no complex mocking needed).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chroma_store(tmp_path: Path):
    """Create a real ChromaStore backed by a temp directory."""
    persist_dir = str(tmp_path / "chroma")

    # Patch get_settings so ChromaStore uses our temp dir
    mock_cfg = MagicMock()
    mock_cfg.chroma_persist_dir = persist_dir
    mock_cfg.chroma_collection_name = "test_experiments"

    with patch("app.memory.chroma_store.get_settings", return_value=mock_cfg):
        from app.memory.chroma_store import ChromaStore
        store = ChromaStore.__new__(ChromaStore)
        store.settings = mock_cfg
        store.collection_name = "test_experiments"
        store._client = None
        store._collection = None
        return store


# ── ChromaStore tests ─────────────────────────────────────────────────────────

def test_chroma_store_empty_initially(tmp_path):
    store = _make_chroma_store(tmp_path)
    assert store.count() == 0


def test_chroma_store_add_and_count(tmp_path):
    store = _make_chroma_store(tmp_path)
    store.add(
        doc_id="exp-001",
        document="Classification with logistic regression, accuracy 0.91",
        metadata={"task_type": "binary_classification"},
        embedding=None,
    )
    assert store.count() == 1


def test_chroma_store_get_all(tmp_path):
    store = _make_chroma_store(tmp_path)
    for i in range(3):
        store.add(
            doc_id=f"exp-{i:03d}",
            document=f"Experiment {i}: regression, RMSE=0.{i}5",
            metadata={"i": i},
            embedding=None,
        )
    all_docs = store.get_all()
    assert len(all_docs) == 3


def test_chroma_store_query_returns_results(tmp_path):
    store = _make_chroma_store(tmp_path)
    store.add(
        doc_id="exp-q",
        document="Binary classification on tabular data with XGBoost, ROC-AUC 0.95",
        metadata={"task": "binary_classification"},
        embedding=None,
    )
    results = store.query("classification XGBoost", n_results=1)
    assert len(results) >= 1
    assert "document" in results[0]


# ── ExperimentMemory tests ────────────────────────────────────────────────────

def _make_experiment_memory(tmp_path: Path):
    """Create ExperimentMemory with real ChromaDB in temp dir."""
    persist_dir = str(tmp_path / "chroma_em")

    mock_cfg = MagicMock()
    mock_cfg.chroma_persist_dir = persist_dir
    mock_cfg.chroma_collection_name = "test_experiments"
    mock_cfg.embedding_model = "all-MiniLM-L6-v2"

    with patch("app.memory.chroma_store.get_settings", return_value=mock_cfg), \
         patch("app.memory.embeddings.get_settings", return_value=mock_cfg):
        from app.memory.experiment_memory import ExperimentMemory
        mem = ExperimentMemory.__new__(ExperimentMemory)
        # Manually initialise without calling __init__ to control settings
        from app.memory.chroma_store import ChromaStore
        store = ChromaStore.__new__(ChromaStore)
        store.settings = mock_cfg
        store.collection_name = "test_experiments"
        store._client = None
        store._collection = None
        mem.store = store
        # Stub embed to return a simple fixed-length vector
        mem.embedder = MagicMock()
        mem.embedder.embed.return_value = [0.1] * 384
        return mem


def test_experiment_memory_store_and_retrieve(tmp_path):
    mem = _make_experiment_memory(tmp_path)
    mem.store.add(
        doc_id="mem-test-001",
        document=(
            "Dataset: titanic.csv (891 rows). Task: binary_classification. "
            "Best model: RandomForest. Accuracy: 0.83."
        ),
        metadata={"experiment_id": "mem-test-001", "task_type": "binary_classification"},
        embedding=None,
    )
    count = mem.store.count()
    assert count >= 1

    results = mem.store.query("binary classification", n_results=3)
    assert len(results) >= 1


def test_experiment_memory_empty_returns_empty(tmp_path):
    mem = _make_experiment_memory(tmp_path)
    # Empty store — query returns empty list (not an error)
    results = mem.store.query("anything", n_results=3)
    # ChromaDB returns empty when n_results > count, handled by min(n_results, count)
    assert isinstance(results, list)


# ── RAG Retriever tests ───────────────────────────────────────────────────────

def test_rag_retriever_initialises(tmp_path):
    persist_dir = str(tmp_path / "chroma_rag")
    mock_cfg = MagicMock()
    mock_cfg.chroma_persist_dir = persist_dir
    mock_cfg.chroma_collection_name = "test_experiments"
    mock_cfg.embedding_model = "all-MiniLM-L6-v2"

    # ExperimentMemory does NOT call get_settings — only its sub-components do
    with patch("app.memory.chroma_store.get_settings", return_value=mock_cfg), \
         patch("app.memory.embeddings.get_settings", return_value=mock_cfg):
        from app.rag.retrieval import RAGRetriever
        r = RAGRetriever()
        assert isinstance(r, RAGRetriever)


def test_rag_retriever_safe_on_none_memory():
    """RAGRetriever never raises — degrades to empty results when memory=None."""
    from app.rag.retrieval import RAGRetriever
    r = RAGRetriever.__new__(RAGRetriever)
    r.n_results = 5
    r.memory = None  # Simulate broken memory
    result = r._safe_retrieve("any query")
    assert result == []


def test_rag_retriever_context_methods_return_strings():
    """All three retrieval modes must return a non-empty string."""
    from app.rag.retrieval import RAGRetriever
    r = RAGRetriever.__new__(RAGRetriever)
    r.n_results = 5
    r.memory = None  # No memory → fallback strings returned

    ctx1 = r.retrieve_for_model_selection("tabular dataset", "regression")
    ctx2 = r.retrieve_for_preprocessing("tabular dataset with missing values")
    ctx3 = r.retrieve_for_chat("What is the best model for this dataset?")

    assert isinstance(ctx1, str) and len(ctx1) > 0
    assert isinstance(ctx2, str) and len(ctx2) > 0
    assert isinstance(ctx3, str) and len(ctx3) > 0
