"""
AutonomDS Agent State
======================
Central TypedDict that flows through the entire LangGraph workflow.
Every agent reads from and writes to this shared state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from typing_extensions import TypedDict


# ── Enumerations ──────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REFLECTING = "reflecting"


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"


class PipelineStage(str, Enum):
    INGESTION = "ingestion"
    EDA = "eda"
    CLEANING = "cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    TRAINING = "training"
    EVALUATION = "evaluation"
    EXPLAINABILITY = "explainability"
    REPORT = "report"
    MEMORY = "memory"
    COMPLETE = "complete"


# ── Sub-state Models ──────────────────────────────────────────────────────────

@dataclass
class AgentExecutionRecord:
    """Record of a single agent's execution."""
    agent_name: str
    stage: PipelineStage
    status: AgentStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0
    confidence: float = 1.0
    observations: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "stage": self.stage.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "retry_count": self.retry_count,
            "confidence": self.confidence,
            "observations": self.observations,
            "actions_taken": self.actions_taken,
        }


@dataclass
class DatasetInfo:
    """Metadata about the uploaded dataset."""
    filename: str
    file_path: str
    extension: str
    checksum: str
    n_rows: int = 0
    n_cols: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    target_column: Optional[str] = None
    task_type: TaskType = TaskType.UNKNOWN
    memory_mb: float = 0.0
    upload_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "extension": self.extension,
            "checksum": self.checksum,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "target_column": self.target_column,
            "task_type": self.task_type.value,
            "memory_mb": self.memory_mb,
            "upload_timestamp": self.upload_timestamp,
        }


@dataclass
class EDAResults:
    """Results from the EDA agent."""
    missing_summary: dict[str, Any] = field(default_factory=dict)
    correlation_matrix: Optional[list[list[float]]] = None
    outlier_summary: dict[str, Any] = field(default_factory=dict)
    class_balance: dict[str, Any] = field(default_factory=dict)
    skewness: dict[str, float] = field(default_factory=dict)
    chart_paths: list[str] = field(default_factory=list)
    nl_insights: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelResult:
    """Result for a single trained model."""
    model_name: str
    model_type: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    training_time_seconds: float = 0.0
    model_path: Optional[str] = None
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "params": self.params,
            "metrics": self.metrics,
            "cv_scores": self.cv_scores,
            "cv_mean": sum(self.cv_scores) / len(self.cv_scores) if self.cv_scores else 0,
            "training_time_seconds": self.training_time_seconds,
            "model_path": self.model_path,
            "rank": self.rank,
        }


# ── Central Agent State ───────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    Central state TypedDict for the LangGraph pipeline.

    This is the single source of truth passed between all agents.
    Every key is optional (total=False) since agents incrementally populate it.
    """

    # ── Identity ────────────────────────────────────────────────────────
    experiment_id: str
    run_id: str
    created_at: str
    current_stage: str          # PipelineStage value
    current_agent: str

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset_info: dict[str, Any]          # DatasetInfo.to_dict()
    raw_data_path: str                     # Path to original uploaded file
    processed_data_path: str              # Path to cleaned/processed CSV

    # ── Target / Task ───────────────────────────────────────────────────
    target_column: str
    task_type: str                         # TaskType value
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]

    # ── EDA Results ─────────────────────────────────────────────────────
    eda_results: dict[str, Any]
    eda_charts: list[str]                  # File paths to saved charts
    eda_insights: list[str]               # NL insights from LLM
    eda_warnings: list[str]

    # ── Cleaning ────────────────────────────────────────────────────────
    cleaning_report: dict[str, Any]
    cleaning_actions: list[str]

    # ── Feature Engineering ─────────────────────────────────────────────
    feature_report: dict[str, Any]
    selected_features: list[str]
    n_features_original: int
    n_features_selected: int

    # ── Model Selection ─────────────────────────────────────────────────
    candidate_models: list[str]
    baseline_results: dict[str, Any]
    selected_model_types: list[str]

    # ── Training ────────────────────────────────────────────────────────
    trained_models: list[dict[str, Any]]  # List of ModelResult.to_dict()
    best_model_name: str
    best_model_path: str
    training_duration_seconds: float

    # ── Evaluation ──────────────────────────────────────────────────────
    evaluation_results: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    confusion_matrix: list[list[int]]
    roc_auc: float

    # ── Explainability ──────────────────────────────────────────────────
    shap_values_path: str
    feature_importance: dict[str, float]
    explainability_report: str

    # ── Reports ─────────────────────────────────────────────────────────
    pdf_report_path: str
    markdown_report_path: str
    report_summary: str

    # ── Memory ──────────────────────────────────────────────────────────
    similar_experiments: list[dict[str, Any]]
    memory_stored: bool

    # ── Execution Tracking ──────────────────────────────────────────────
    agent_executions: list[dict[str, Any]]   # AgentExecutionRecord.to_dict()
    messages: list[dict[str, str]]            # Chat-style message history
    errors: list[str]
    retry_count: int

    # ── Orchestrator Control ─────────────────────────────────────────────
    should_reflect: bool                  # Trigger reflection loop
    reflection_notes: list[str]          # Critique from reflection
    confidence_score: float              # Overall pipeline confidence (0-1)
    pipeline_complete: bool
