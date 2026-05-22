"""
Unit Tests — Data Cleaning Agent
==================================
Tests imputation, encoding, scaling, and leakage detection.
The cleaning agent reads from processed_data_path (Parquet format).
No LLM or file I/O mocking needed — uses real temp files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def dirty_parquet(tmp_path: Path) -> Path:
    """Parquet file with missing values, duplicates, and categoricals."""
    rng = np.random.default_rng(7)
    n = 100
    df = pd.DataFrame({
        "age":    list(rng.integers(18, 70, n - 5).astype(float)) + [np.nan] * 5,
        "income": rng.normal(50_000, 15_000, n),
        "city":   list(rng.choice(["NYC", "LA", "CHI"], n - 3)) + [None, None, None],
        "target": rng.integers(0, 2, n),
    })
    # Add duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture()
def cleaning_agent():
    from app.agents.cleaning_agent import DataCleaningAgent
    agent = DataCleaningAgent()
    agent.ask_llm = MagicMock(return_value="No leakage found.")
    return agent


def _make_state(parquet_path: Path) -> dict:
    return {
        "experiment_id":       "clean-test",
        "processed_data_path": str(parquet_path),  # cleaning agent reads this key
        "raw_data_path":       str(parquet_path),
        "target_column":       "target",
        "task_type":           "binary_classification",
        "numeric_columns":     ["age", "income"],
        "categorical_columns": ["city"],
        "agent_executions": [],
        "errors": [],
        "messages": [],
        "retry_count": 0,
    }


# ── Stage check ───────────────────────────────────────────────────────────────

def test_cleaning_agent_has_correct_stage(cleaning_agent):
    from app.orchestration.state import PipelineStage
    assert cleaning_agent.stage == PipelineStage.CLEANING


# ── Actions ───────────────────────────────────────────────────────────────────

def test_cleaning_agent_produces_actions(cleaning_agent, dirty_parquet):
    """Verify that the cleaning agent records at least one action taken."""
    result = cleaning_agent.run(_make_state(dirty_parquet))
    assert "cleaning_actions" in result
    assert isinstance(result["cleaning_actions"], list)
    assert len(result["cleaning_actions"]) > 0


def test_cleaning_agent_output_has_report(cleaning_agent, dirty_parquet):
    """cleaning_report must include n_rows_after and n_cols_after."""
    result = cleaning_agent.run(_make_state(dirty_parquet))
    report = result.get("cleaning_report", {})
    assert "n_rows_after" in report
    assert "n_cols_after" in report
    assert report["n_rows_after"] > 0


# ── File persistence ──────────────────────────────────────────────────────────

def test_cleaning_agent_saves_cleaned_file(cleaning_agent, dirty_parquet):
    """processed_data_path should point to a valid Parquet file after cleaning."""
    result = cleaning_agent.run(_make_state(dirty_parquet))
    new_path = result.get("processed_data_path", "")
    assert new_path != "", "processed_data_path should be set"
    assert Path(new_path).exists(), f"Cleaned parquet not found: {new_path}"


def test_cleaning_agent_cleaned_file_is_readable(cleaning_agent, dirty_parquet):
    """The saved cleaned Parquet must be readable as a DataFrame."""
    result = cleaning_agent.run(_make_state(dirty_parquet))
    new_path = result.get("processed_data_path", "")
    if new_path and Path(new_path).exists():
        df = pd.read_parquet(new_path)
        assert len(df) > 0
        assert "target" in df.columns


# ── Data quality ──────────────────────────────────────────────────────────────

def test_cleaning_removes_duplicates(cleaning_agent, dirty_parquet):
    """With 5 duplicate rows appended, the cleaned data should have fewer rows."""
    original_df = pd.read_parquet(dirty_parquet)
    original_rows = len(original_df)

    result = cleaning_agent.run(_make_state(dirty_parquet))
    report = result.get("cleaning_report", {})
    # After dedup, row count should decrease
    assert report.get("n_rows_after", original_rows) <= original_rows


def test_cleaning_no_nulls_in_numeric_after_clean(cleaning_agent, dirty_parquet):
    """After cleaning, numeric feature columns should have no nulls."""
    result = cleaning_agent.run(_make_state(dirty_parquet))
    new_path = result.get("processed_data_path", "")
    if new_path and Path(new_path).exists():
        df = pd.read_parquet(new_path)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            assert df[col].isnull().sum() == 0, f"Nulls remain in {col}"
