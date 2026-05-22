"""
Data Ingestion Agent
=====================
Handles loading of CSV, Excel, Parquet, SQLite, and Kaggle datasets.
Performs schema validation, dtype inference, target column detection,
and produces a normalized DataFrame for the rest of the pipeline.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
from datetime import timezone, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage, TaskType
from app.utils.config import get_settings
from app.utils.helpers import (
    compute_dataset_stats,
    detect_target_column,
    get_column_types,
    infer_task_type,
    now_iso,
)
from app.utils.validators import FileValidator, ValidationError


class DataIngestionAgent(BaseAgent):
    """
    Responsible for loading, validating, and normalizing datasets.

    Supports:
    - CSV files (auto-detects encoding, delimiter)
    - Excel files (.xlsx, .xls)
    - Parquet files
    - SQLite databases (first table or specified table)
    - Kaggle datasets (via kaggle API)

    Output:
    - Saves processed DataFrame to disk as Parquet
    - Populates dataset_info, feature_columns, target_column, task_type in state
    """

    name = "data_ingestion_agent"
    description = "Loads, validates, and normalizes uploaded datasets"
    stage = PipelineStage.INGESTION
    max_retries = 1

    def __init__(self, model_override: Optional[str] = None) -> None:
        super().__init__(model_override)
        self.validator = FileValidator()

    def execute(self, state: AgentState) -> AgentState:
        """Main ingestion logic."""
        raw_path = state.get("raw_data_path")
        if not raw_path:
            raise AgentError("No raw_data_path in state. Upload a file first.")

        path = Path(raw_path)
        if not path.exists():
            raise AgentError(f"File not found: {raw_path}")

        ext = path.suffix.lstrip(".").lower()
        self.log_action("loading_dataset", path=str(path), ext=ext)

        # ── Load DataFrame ────────────────────────────────────────────────
        df = self._load_by_extension(path, ext, state)
        self.log_action("dataset_loaded", rows=len(df), cols=len(df.columns))

        # ── Infer dtypes ─────────────────────────────────────────────────
        df = self._optimize_dtypes(df)

        # ── Detect target column ─────────────────────────────────────────
        target_col = state.get("target_column") or detect_target_column(df)
        if target_col and target_col not in df.columns:
            self.logger.warning("target_column_not_found", target=target_col)
            target_col = detect_target_column(df)

        # ── Infer task type ───────────────────────────────────────────────
        task_type = TaskType.UNKNOWN
        if target_col and target_col in df.columns:
            task_str = infer_task_type(df, target_col)
            task_type = TaskType(task_str)

        # ── Get column categories ─────────────────────────────────────────
        col_types = get_column_types(df)

        feature_cols = [c for c in df.columns if c != target_col]
        stats = compute_dataset_stats(df)

        # ── Save processed data ───────────────────────────────────────────
        settings = get_settings()
        processed_path = settings.upload_dir / f"{path.stem}_processed.parquet"
        df.to_parquet(processed_path, index=False)

        # ── Generate NL summary via LLM ───────────────────────────────────
        summary = self._generate_dataset_summary(df, target_col, task_type, stats)

        # ── Build dataset_info dict ───────────────────────────────────────
        dataset_info = {
            "filename": path.name,
            "file_path": str(path),
            "extension": ext,
            "checksum": state.get("dataset_info", {}).get("checksum", ""),
            "n_rows": stats["n_rows"],
            "n_cols": stats["n_cols"],
            "n_numeric": stats["n_numeric"],
            "n_categorical": stats["n_categorical"],
            "columns": stats["columns"],
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "target_column": target_col,
            "task_type": task_type.value,
            "memory_mb": stats["memory_mb"],
            "n_duplicates": stats["n_duplicates"],
            "total_missing": stats["total_missing"],
            "missing_pct": stats["missing_pct"],
            "upload_timestamp": now_iso(),
            "llm_summary": summary,
        }

        # ── Populate state ─────────────────────────────────────────────────
        state = dict(state)  # type: ignore[assignment]
        state["dataset_info"] = dataset_info
        state["processed_data_path"] = str(processed_path)
        state["target_column"] = target_col or ""
        state["task_type"] = task_type.value
        state["feature_columns"] = feature_cols
        state["categorical_columns"] = col_types["categorical"]
        state["numeric_columns"] = col_types["numeric"]

        self.logger.info(
            "ingestion_complete",
            rows=stats["n_rows"],
            cols=stats["n_cols"],
            target=target_col,
            task_type=task_type.value,
        )
        return state  # type: ignore[return-value]

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_by_extension(
        self, path: Path, ext: str, state: AgentState
    ) -> pd.DataFrame:
        """Dispatch to the correct loader by file extension."""
        loaders = {
            "csv": self._load_csv,
            "xlsx": self._load_excel,
            "xls": self._load_excel,
            "parquet": self._load_parquet,
            "sqlite": self._load_sqlite,
            "db": self._load_sqlite,
        }
        loader = loaders.get(ext)
        if loader is None:
            raise AgentError(f"Unsupported file extension: .{ext}")
        return loader(path, state)

    def _load_csv(self, path: Path, state: AgentState) -> pd.DataFrame:
        """Load CSV with auto-detection of delimiter and encoding."""
        # Try to detect delimiter
        with open(path, "rb") as f:
            sample = f.read(4096).decode("utf-8", errors="replace")

        # Count potential delimiters
        delimiter = ","
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)  # type: ignore[arg-type]
        if counts[best] > counts[","]:
            delimiter = best

        for encoding in ["utf-8", "latin-1", "cp1252", "utf-16"]:
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=delimiter,
                    low_memory=False,
                    on_bad_lines="skip",
                )
                self.log_action("csv_loaded", encoding=encoding, delimiter=repr(delimiter))
                return df
            except UnicodeDecodeError:
                continue
        raise AgentError("Cannot determine CSV encoding.")

    def _load_excel(self, path: Path, state: AgentState) -> pd.DataFrame:
        """Load Excel file, taking first sheet by default."""
        try:
            xf = pd.ExcelFile(path)
            sheet = xf.sheet_names[0]
            df = pd.read_excel(xf, sheet_name=sheet)
            self.log_action("excel_loaded", sheet=sheet, n_sheets=len(xf.sheet_names))
            return df
        except Exception as e:
            raise AgentError(f"Excel loading failed: {e}") from e

    def _load_parquet(self, path: Path, state: AgentState) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            df = pd.read_parquet(path)
            self.log_action("parquet_loaded")
            return df
        except Exception as e:
            raise AgentError(f"Parquet loading failed: {e}") from e

    def _load_sqlite(self, path: Path, state: AgentState) -> pd.DataFrame:
        """Load first table from SQLite database."""
        try:
            conn = sqlite3.connect(path)
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )
            if tables.empty:
                raise AgentError("SQLite database contains no tables.")
            table_name = tables.iloc[0]["name"]
            df = pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)
            conn.close()
            self.log_action("sqlite_loaded", table=table_name)
            return df
        except Exception as e:
            raise AgentError(f"SQLite loading failed: {e}") from e

    # ── dtype Optimization ─────────────────────────────────────────────────────

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce memory usage by downcasting numeric types
        and converting object columns with few unique values to category.
        """
        # Downcast integers
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        # Downcast floats
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Convert low-cardinality strings to category
        for col in df.select_dtypes(include=["object"]).columns:
            n_unique = df[col].nunique()
            if n_unique / len(df) < 0.5 and n_unique < 100:
                df[col] = df[col].astype("category")

        return df

    # ── LLM Summary ───────────────────────────────────────────────────────────

    def _generate_dataset_summary(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        task_type: TaskType,
        stats: dict[str, Any],
    ) -> str:
        """Generate a natural-language summary of the dataset using LLM."""
        # Build a compact schema description
        schema_lines = []
        for col in df.columns[:20]:  # Limit to first 20 cols for token budget
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            n_null = df[col].isnull().sum()
            schema_lines.append(f"  - {col} [{dtype}]: {n_unique} unique, {n_null} nulls")

        schema_str = "\n".join(schema_lines)
        if len(df.columns) > 20:
            schema_str += f"\n  ... and {len(df.columns) - 20} more columns"

        system = (
            "You are an expert data scientist. Given a dataset schema, "
            "provide a concise 3-5 sentence summary describing: "
            "what this dataset likely represents, key characteristics, "
            "and what kind of ML problem it is suited for."
        )
        user = (
            f"Dataset: {stats['n_rows']:,} rows × {stats['n_cols']} columns\n"
            f"Target column: {target_col or 'None detected'}\n"
            f"Task type: {task_type.value}\n"
            f"Missing data: {stats['missing_pct']}%\n"
            f"Memory: {stats['memory_mb']} MB\n\n"
            f"Schema:\n{schema_str}\n\n"
            "Summarize this dataset:"
        )
        return self.ask_llm(system, user)

    def compute_confidence(self, state: AgentState) -> float:
        """Confidence based on how well we could parse the dataset."""
        info = state.get("dataset_info", {})
        if not info:
            return 0.0
        score = 1.0
        # Penalize for high missing rate
        missing_pct = info.get("missing_pct", 0)
        if missing_pct > 50:
            score -= 0.3
        elif missing_pct > 20:
            score -= 0.1
        # Penalize if no target detected
        if not info.get("target_column"):
            score -= 0.2
        return max(0.1, score)

    def _success_message(self, state: AgentState) -> str:
        info = state.get("dataset_info", {})
        return (
            f"✅ Dataset loaded: **{info.get('filename', 'unknown')}** — "
            f"{info.get('n_rows', 0):,} rows × {info.get('n_cols', 0)} columns. "
            f"Task type: **{info.get('task_type', 'unknown')}**. "
            f"Target: **{info.get('target_column', 'None')}**."
        )
