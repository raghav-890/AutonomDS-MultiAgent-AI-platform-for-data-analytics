"""
Integration Tests — LangGraph Pipeline
=========================================
Tests graph compilation, state flow, and conditional routing
using a tiny synthetic dataset. No LLM or Redis required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def tiny_csv(tmp_path_factory) -> str:
    """Write a tiny CSV for end-to-end pipeline testing."""
    tmp = tmp_path_factory.mktemp("pipeline_data")
    path = tmp / "tiny.csv"
    rng = np.random.default_rng(99)
    n = 60
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "cat": rng.choice(["a", "b"], n),
        "target": rng.choice([0, 1], n),
    })
    df.to_csv(path, index=False)
    return str(path)


# ── Graph compilation ─────────────────────────────────────────────────────────

def test_pipeline_graph_compiles():
    """The LangGraph graph should compile without errors."""
    from app.orchestration.graph import build_pipeline_graph
    graph = build_pipeline_graph(use_checkpointing=False)
    assert graph is not None


def test_pipeline_graph_has_all_nodes():
    """All required nodes should be present in the graph."""
    from app.orchestration.graph import build_pipeline_graph
    graph = build_pipeline_graph(use_checkpointing=False)
    # LangGraph compiled graph — check nodes via graph attribute
    nodes = list(graph.get_graph().nodes.keys())
    required = {"ingest", "eda", "clean", "feature_eng", "model_select",
                "train", "evaluate", "reflect", "explain", "report", "memory"}
    for node in required:
        assert node in nodes, f"Missing node: {node}"


# ── Routing logic ─────────────────────────────────────────────────────────────

def test_route_after_ingest_success():
    """If dataset_info is set, route to EDA."""
    from app.orchestration.graph import route_after_ingest
    state = {"dataset_info": {"n_rows": 100}, "errors": []}
    assert route_after_ingest(state) == "eda"


def test_route_after_ingest_failure():
    """If dataset_info is None and errors exist, route to end."""
    from app.orchestration.graph import route_after_ingest
    state = {"dataset_info": None, "errors": ["ingestion failed"]}
    assert route_after_ingest(state) == "end"


def test_route_after_evaluate_to_reflect():
    """Low confidence triggers reflection before explainability."""
    from app.orchestration.graph import route_after_evaluate
    state = {
        "should_reflect": True,
        "retry_count": 0,
        "leaderboard": [{"model_name": "XGB", "metrics": {"f1": 0.3}}],
    }
    assert route_after_evaluate(state) == "reflect"


def test_route_after_evaluate_to_explain():
    """Normal path: evaluation done, proceed to explain."""
    from app.orchestration.graph import route_after_evaluate
    state = {
        "should_reflect": False,
        "retry_count": 0,
        "leaderboard": [{"model_name": "XGB", "metrics": {"f1": 0.85}}],
    }
    assert route_after_evaluate(state) == "explain"


def test_route_after_reflect_retries_training():
    """After reflection with retry_count < 3, go back to train."""
    from app.orchestration.graph import route_after_reflect
    state = {"retry_count": 1}
    assert route_after_reflect(state) == "train"


def test_route_after_reflect_ends_on_max_retries():
    """After 3 retries, reflection routes to END."""
    from app.orchestration.graph import route_after_reflect
    state = {"retry_count": 3}
    assert route_after_reflect(state) == "end"


# ── State initialisation ──────────────────────────────────────────────────────

def test_agent_state_defaults():
    """AgentState TypedDict should accept partial state with defaults."""
    from app.orchestration.state import AgentState
    state: AgentState = {
        "experiment_id": "test-123",
        "errors": [],
        "agent_executions": [],
        "messages": [],
        "retry_count": 0,
        "should_reflect": False,
        "pipeline_complete": False,
    }
    assert state["experiment_id"] == "test-123"
    assert state["errors"] == []
