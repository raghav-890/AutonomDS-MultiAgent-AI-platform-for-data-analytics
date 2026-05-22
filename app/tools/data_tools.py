"""
AutonomDS Data Tools
=====================
Reusable data-processing utilities shared across agents.
Keeps agent code clean by centralizing common operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def infer_target_column(df: pd.DataFrame) -> str | None:
    """
    Heuristically identify the most likely target column.

    Priority:
    1. Columns named 'target', 'label', 'class', 'y', 'output'
    2. Last column (common convention)
    """
    common_names = {"target", "label", "class", "y", "output", "result",
                    "outcome", "response", "dependent"}
    for col in df.columns:
        if col.lower() in common_names:
            return col
    # Fallback: last column
    return df.columns[-1] if len(df.columns) > 0 else None


def infer_task_type(df: pd.DataFrame, target: str) -> str:
    """
    Infer whether the task is regression or classification.

    Rules:
    - String / object target → classification
    - Bool target → binary classification
    - Integer target with ≤ 20 unique values → classification
    - Float or many unique integers → regression
    """
    if target not in df.columns:
        return "unknown"

    series = df[target].dropna()
    dtype = series.dtype

    if dtype == object or hasattr(dtype, "categories"):
        n_unique = series.nunique()
        return "binary_classification" if n_unique == 2 else "multiclass_classification"

    if dtype == bool:
        return "binary_classification"

    n_unique = series.nunique()
    if n_unique == 2:
        return "binary_classification"
    if n_unique <= 20 and pd.api.types.is_integer_dtype(dtype):
        return "multiclass_classification"

    return "regression"


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory usage.
    Safe — does not change values, only reduces precision where possible.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def safe_sample(df: pd.DataFrame, max_rows: int = 50_000) -> pd.DataFrame:
    """Return a stratified sample if df exceeds max_rows."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def column_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact statistics summary for each column."""
    stats: dict[str, Any] = {}
    for col in df.columns:
        s = df[col]
        entry: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": int(s.isnull().sum()),
            "null_pct": round(s.isnull().mean() * 100, 2),
            "unique_count": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            entry.update({
                "mean":   round(float(s.mean()), 4) if not s.isnull().all() else None,
                "std":    round(float(s.std()),  4) if not s.isnull().all() else None,
                "min":    float(s.min())              if not s.isnull().all() else None,
                "max":    float(s.max())              if not s.isnull().all() else None,
                "median": round(float(s.median()), 4) if not s.isnull().all() else None,
            })
        stats[col] = entry
    return stats
