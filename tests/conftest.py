"""
Test fixtures and shared utilities.
"""

import io
import os
import pytest
import numpy as np
import pandas as pd

# Point to test env
os.environ.update({
    "APP_ENV": "testing",
    "DEBUG": "true",
    "DATABASE_URL": "sqlite+aiosqlite:///./test_autonomds.db",
    "MLFLOW_TRACKING_URI": "sqlite:///./test_mlflow.db",
    "CHROMA_PERSIST_DIR": "./test_chroma",
    "UPLOAD_DIR": "./test_uploads",
    "REPORTS_DIR": "./test_reports",
    "LLM_PROVIDER": "huggingface",
    "HUGGINGFACE_API_TOKEN": "dummy_token_for_tests",
})


@pytest.fixture(scope="session")
def sample_classification_df() -> pd.DataFrame:
    """A small binary classification dataset."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.choice(["a", "b", "c"], n),
        "age": np.random.randint(18, 80, n),
        "income": np.random.exponential(50000, n),
        "target": np.random.randint(0, 2, n),
    })


@pytest.fixture(scope="session")
def sample_regression_df() -> pd.DataFrame:
    """A small regression dataset."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 4)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5
    return pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"]) | pd.DataFrame({"price": y})


@pytest.fixture(scope="session")
def sample_csv_bytes(sample_classification_df) -> bytes:
    """CSV bytes of the classification dataset."""
    buf = io.BytesIO()
    sample_classification_df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture(scope="session")
def base_state(tmp_path_factory, sample_classification_df) -> dict:
    """A minimal AgentState with a dataset saved to disk."""
    import tempfile
    from pathlib import Path

    tmp = tmp_path_factory.mktemp("data")
    csv_path = tmp / "test_data.csv"
    sample_classification_df.to_csv(csv_path, index=False)
    parquet_path = tmp / "test_data.parquet"
    sample_classification_df.to_parquet(parquet_path, index=False)

    return {
        "experiment_id": "test_exp_001",
        "run_id": "test_run_001",
        "created_at": "2024-01-01T00:00:00+00:00",
        "raw_data_path": str(csv_path),
        "processed_data_path": str(parquet_path),
        "target_column": "target",
        "task_type": "binary_classification",
        "feature_columns": ["feature_1", "feature_2", "feature_3", "age", "income"],
        "categorical_columns": ["feature_3"],
        "numeric_columns": ["feature_1", "feature_2", "age", "income"],
        "current_stage": "ingestion",
        "current_agent": "",
        "agent_executions": [],
        "messages": [],
        "errors": [],
        "retry_count": 0,
        "should_reflect": False,
        "reflection_notes": [],
        "confidence_score": 1.0,
        "pipeline_complete": False,
        "dataset_info": {
            "filename": "test_data.csv",
            "n_rows": 200,
            "n_cols": 6,
            "target_column": "target",
            "task_type": "binary_classification",
            "missing_pct": 0.0,
        },
    }
