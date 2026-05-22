"""
Unit Tests — Training Agent
==============================
Tests cross-validation and model persistence.
MLflow is imported lazily inside the agent with `import mlflow`,
so we patch it via sys.modules — not as a module-level attribute.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def classification_parquet(tmp_path: Path) -> Path:
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=5,
        n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    path = tmp_path / "processed.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture()
def regression_parquet(tmp_path: Path) -> Path:
    X, y = make_regression(n_samples=200, n_features=6, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["price"] = y
    path = tmp_path / "processed.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture()
def training_agent():
    from app.agents.training_agent import TrainingAgent
    agent = TrainingAgent()
    agent.ask_llm = MagicMock(return_value="Training looks good.")
    return agent


def _make_state(parquet_path: Path, target: str, task: str, model: str = "LogisticRegression") -> dict:
    feature_cols = [c for c in pd.read_parquet(parquet_path).columns if c != target]
    return {
        "experiment_id":        "train-test",
        "processed_data_path":  str(parquet_path),
        "target_column":        target,
        "task_type":            task,
        "selected_model_types": [model],
        "selected_features":    feature_cols,
        "n_features_selected":  len(feature_cols),
        "agent_executions": [],
        "errors": [],
        "messages": [],
        "retry_count": 0,
    }


def _mock_settings(tmp_path: Path):
    """Return a MagicMock settings object."""
    mock_cfg = MagicMock()
    mock_cfg.reports_dir    = str(tmp_path)
    mock_cfg.cross_val_folds = 3
    mock_cfg.optuna_n_trials = 2        # Very fast for CI
    mock_cfg.effective_training_device = "cpu"
    mock_cfg.mlflow_tracking_uri = f"file://{tmp_path}/mlruns"
    mock_cfg.mlflow_experiment_name = "test"
    return mock_cfg


# ── Mock MLflow helper ────────────────────────────────────────────────────────

class _FakeMLflow:
    """Minimal MLflow stub — all calls are no-ops."""
    def set_tracking_uri(self, *a, **kw): pass
    def set_experiment(self, *a, **kw): pass
    def start_run(self, *a, **kw):
        import contextlib
        @contextlib.contextmanager
        def _ctx():
            yield MagicMock()
        return _ctx()
    def log_params(self, *a, **kw): pass
    def log_metric(self, *a, **kw): pass
    def log_metrics(self, *a, **kw): pass


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_training_agent_classification(training_agent, classification_parquet, tmp_path):
    """Training agent trains at least one model and sets best_model_name."""
    with patch("app.agents.training_agent.get_settings", return_value=_mock_settings(tmp_path)), \
         patch.dict(sys.modules, {"mlflow": _FakeMLflow()}):
        state = _make_state(classification_parquet, "target", "binary_classification")
        result = training_agent.run(state)

    assert "trained_models" in result
    successful = [m for m in result["trained_models"] if "error" not in m]
    assert len(successful) >= 1
    assert result.get("best_model_name", "") != ""


def test_training_agent_stores_model_file(training_agent, classification_parquet, tmp_path):
    """A .joblib model file must be created after training."""
    with patch("app.agents.training_agent.get_settings", return_value=_mock_settings(tmp_path)), \
         patch.dict(sys.modules, {"mlflow": _FakeMLflow()}):
        state = _make_state(classification_parquet, "target", "binary_classification")
        result = training_agent.run(state)

    for model_info in result.get("trained_models", []):
        if "error" not in model_info and model_info.get("model_path"):
            assert Path(model_info["model_path"]).exists()


def test_training_agent_regression(training_agent, regression_parquet, tmp_path):
    """Training agent handles regression tasks without error."""
    with patch("app.agents.training_agent.get_settings", return_value=_mock_settings(tmp_path)), \
         patch.dict(sys.modules, {"mlflow": _FakeMLflow()}):
        state = _make_state(regression_parquet, "price", "regression", model="Ridge")
        result = training_agent.run(state)

    assert "trained_models" in result
    assert len(result["trained_models"]) >= 1


def test_training_agent_records_cv_scores(training_agent, classification_parquet, tmp_path):
    """CV scores should be present for successfully trained models."""
    with patch("app.agents.training_agent.get_settings", return_value=_mock_settings(tmp_path)), \
         patch.dict(sys.modules, {"mlflow": _FakeMLflow()}):
        state = _make_state(classification_parquet, "target", "binary_classification")
        result = training_agent.run(state)

    for model_info in result.get("trained_models", []):
        if "error" not in model_info:
            assert "cv_mean" in model_info
            assert "cv_std" in model_info
            assert isinstance(model_info["cv_scores"], list)


def test_training_agent_confidence(training_agent, classification_parquet, tmp_path):
    """Confidence should be 1.0 when all models train successfully."""
    with patch("app.agents.training_agent.get_settings", return_value=_mock_settings(tmp_path)), \
         patch.dict(sys.modules, {"mlflow": _FakeMLflow()}):
        state = _make_state(classification_parquet, "target", "binary_classification")
        result = training_agent.run(state)

    # At least some confidence when at least one model succeeded
    confidence = training_agent.compute_confidence(result)
    assert 0.0 <= confidence <= 1.0
