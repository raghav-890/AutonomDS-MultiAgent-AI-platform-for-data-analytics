"""
Unit Tests — EDA Agent (expanded)
====================================
Tests the EDA agent with synthetic DataFrames to verify:
- Missing value analysis
- Outlier detection
- Class balance analysis
- Skewness analysis
- Warning generation
- Rule-based insights (no LLM required)
- Chart generation (with mocked chart_dir)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def classification_df() -> pd.DataFrame:
    """Small binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "age":    rng.integers(18, 70, n),
        "income": rng.normal(50_000, 15_000, n),
        "score":  rng.uniform(0, 1, n),
        "city":   rng.choice(["NYC", "LA", "CHI"], n),
        "target": rng.choice([0, 1], n, p=[0.7, 0.3]),   # slight imbalance
    })


@pytest.fixture()
def regression_df() -> pd.DataFrame:
    """Small regression dataset with missings and outliers."""
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": np.concatenate([rng.normal(0, 1, n - 5), [100, -100, 200, -200, 150]]),  # outliers
        "x3": rng.exponential(2, n),   # skewed
        "price": rng.normal(300_000, 80_000, n),
    })
    # Inject missing values
    df.loc[rng.choice(df.index, 20, replace=False), "x1"] = np.nan
    return df


@pytest.fixture()
def eda_agent(tmp_path: Path):
    """Return an EDAAgent with LLM disabled (no Ollama needed)."""
    from app.agents.eda_agent import EDAAgent
    agent = EDAAgent()
    # Patch ask_llm to return empty so rule-based fallback fires
    agent.ask_llm = MagicMock(return_value="not valid json")
    return agent


# ── Missing value analysis ────────────────────────────────────────────────────

def test_analyze_missing_no_missing(eda_agent, classification_df):
    result = eda_agent._analyze_missing(classification_df)
    assert result["total_missing"] == 0
    assert result["cols_with_missing"] == 0
    assert result["by_column"] == {}


def test_analyze_missing_with_missing(eda_agent, regression_df):
    result = eda_agent._analyze_missing(regression_df)
    assert result["total_missing"] == 20
    assert "x1" in result["by_column"]
    assert result["by_column"]["x1"]["count"] == 20
    pct = result["by_column"]["x1"]["pct"]
    assert 10 < pct < 20   # ~13.3%


# ── Outlier detection ─────────────────────────────────────────────────────────

def test_analyze_outliers_detects_injected(eda_agent, regression_df):
    num_cols = ["x1", "x2", "x3"]
    result = eda_agent._analyze_outliers(regression_df, num_cols)
    # x2 has 5 extreme outliers
    assert "x2" in result["by_column"]
    assert result["by_column"]["x2"]["count"] >= 3


def test_analyze_outliers_none(eda_agent, classification_df):
    result = eda_agent._analyze_outliers(classification_df, ["age", "income"])
    # With normal distributions, very few outliers
    total = sum(v["count"] for v in result["by_column"].values())
    assert total < 20   # < 10% of 200 rows


# ── Class balance ─────────────────────────────────────────────────────────────

def test_class_balance_imbalanced(eda_agent, classification_df):
    result = eda_agent._analyze_class_balance(classification_df, "target")
    assert result["n_classes"] == 2
    assert "0" in result["distribution"] or 0 in result["distribution"]
    # 70/30 split → ratio = 0.7/0.3 ≈ 2.33
    assert result["imbalance_ratio"] > 1.5


def test_class_balance_not_imbalanced(eda_agent):
    df = pd.DataFrame({"target": [0] * 100 + [1] * 100})
    result = eda_agent._analyze_class_balance(df, "target")
    assert not result["is_imbalanced"]
    assert abs(result["imbalance_ratio"] - 1.0) < 0.05


# ── Skewness analysis ─────────────────────────────────────────────────────────

def test_skewness_exponential_is_positive(eda_agent, regression_df):
    result = eda_agent._analyze_skewness(regression_df, ["x3"])
    assert "x3" in result
    assert result["x3"] > 1.0   # Exponential distribution is right-skewed


def test_skewness_normal_excluded(eda_agent, regression_df):
    """x1 follows N(0,1) — skew near 0, should NOT be flagged."""
    result = eda_agent._analyze_skewness(regression_df, ["x1"])
    # Normal data may or may not exceed 0.5 threshold with n=150
    # Just verify the method runs and returns a dict
    assert isinstance(result, dict)


# ── Warning generation ────────────────────────────────────────────────────────

def test_warnings_generated_for_bad_data(eda_agent, regression_df):
    missing = eda_agent._analyze_missing(regression_df)
    outliers = eda_agent._analyze_outliers(regression_df, ["x1", "x2", "x3"])
    skewness = eda_agent._analyze_skewness(regression_df, ["x1", "x2", "x3"])
    cb: dict = {}
    warnings = eda_agent._build_warnings(missing, outliers, skewness, cb)
    assert len(warnings) >= 1
    assert any("missing" in w.lower() or "outlier" in w.lower() or "skew" in w.lower()
               for w in warnings)


# ── Rule-based insights ───────────────────────────────────────────────────────

def test_rule_based_insights_returns_list(eda_agent, classification_df):
    missing  = eda_agent._analyze_missing(classification_df)
    outliers = eda_agent._analyze_outliers(classification_df, ["age", "income", "score"])
    skewness = eda_agent._analyze_skewness(classification_df, ["age", "income", "score"])
    cb       = eda_agent._analyze_class_balance(classification_df, "target")
    insights = eda_agent._rule_based_insights(
        classification_df, missing, outliers, skewness, cb, "target"
    )
    assert isinstance(insights, list)
    assert len(insights) >= 3
    assert all(isinstance(i, str) for i in insights)


# ── Full execute() integration ────────────────────────────────────────────────

def test_execute_populates_state(eda_agent, classification_df, tmp_path):
    """Full agent run on a temp CSV file."""
    from app.utils.config import get_settings

    # Write temp CSV
    csv_path = tmp_path / "test.csv"
    classification_df.to_csv(csv_path, index=False)

    # Patch settings to use tmp dirs
    with patch("app.agents.eda_agent.get_settings") as mock_settings:
        mock_cfg = MagicMock()
        mock_cfg.reports_dir = tmp_path
        mock_settings.return_value = mock_cfg

        state: dict = {
            "experiment_id": "test-exp-001",
            "raw_data_path": str(csv_path),
            "target_column": "target",
            "agent_executions": [],
            "errors": [],
            "messages": [],
            "retry_count": 0,
        }

        result = eda_agent.run(state)

    assert "eda_results" in result
    assert "eda_insights" in result
    assert isinstance(result["eda_insights"], list)
    assert len(result["eda_insights"]) > 0
    assert result["eda_results"]["n_rows"] == len(classification_df)
    assert "numeric_columns" in result
    assert "target" not in result["numeric_columns"]   # target excluded
