"""
AutonomDS Utility Helpers
==========================
Common utility functions used across the entire platform.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd


# ── ID Generation ─────────────────────────────────────────────────────────────

def generate_experiment_id() -> str:
    """Generate a unique experiment ID (timestamp + short UUID)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"exp_{ts}_{short_uuid}"


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return uuid.uuid4().hex


# ── Timing ────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = "") -> Generator[dict[str, float], None, None]:
    """
    Context manager that measures elapsed time.

    Usage::

        with timer("training") as t:
            model.fit(X, y)
        print(f"Took {t['elapsed']:.2f}s")
    """
    result: dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


# ── DataFrame Utilities ───────────────────────────────────────────────────────

def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Infer ML task type from the target column.

    Returns:
        'binary_classification' | 'multiclass_classification' | 'regression'
    """
    series = df[target_col].dropna()

    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique == 2:
            return "binary_classification"
        if n_unique <= 20 and series.dtype in (int, "int64", "int32"):
            return "multiclass_classification"
        return "regression"
    else:
        n_unique = series.nunique()
        if n_unique == 2:
            return "binary_classification"
        return "multiclass_classification"


def detect_target_column(df: pd.DataFrame) -> str | None:
    """
    Heuristically detect the most likely target column.
    Prioritizes columns named 'target', 'label', 'y', 'survived', etc.
    """
    TARGET_NAMES = {
        "target", "label", "labels", "y", "output", "class",
        "survived", "churn", "default", "fraud", "price",
        "salary", "outcome", "result", "category",
    }
    lower_cols = {col.lower(): col for col in df.columns}
    for name in TARGET_NAMES:
        if name in lower_cols:
            return lower_cols[name]
    # Last column as fallback (common convention)
    return df.columns[-1]


def get_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Categorize columns by dtype."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    boolean = df.select_dtypes(include=["bool"]).columns.tolist()
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "boolean": boolean,
    }


def compute_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute high-level dataset statistics."""
    col_types = get_column_types(df)
    missing = df.isnull().sum()
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_numeric": len(col_types["numeric"]),
        "n_categorical": len(col_types["categorical"]),
        "n_datetime": len(col_types["datetime"]),
        "total_missing": int(missing.sum()),
        "missing_pct": round(missing.sum() / (len(df) * len(df.columns)) * 100, 2),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "n_duplicates": int(df.duplicated().sum()),
        "columns": df.columns.tolist(),
    }


def safe_json_serialize(obj: Any) -> Any:
    """Recursively make objects JSON-serializable."""
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_serialize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def truncate_string(s: str, max_len: int = 500) -> str:
    """Truncate long strings for logging/display."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... [{len(s) - max_len} chars truncated]"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def now_utc() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return now_utc().isoformat()
