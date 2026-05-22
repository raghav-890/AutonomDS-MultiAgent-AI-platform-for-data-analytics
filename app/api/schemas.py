"""
Pydantic API Schemas
=====================
Request/Response models for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    experiment_id: str
    filename: str
    checksum: str
    n_rows: int
    n_cols: int
    columns: list[str]
    detected_target: Optional[str] = None
    task_type: str
    file_path: str
    message: str = "File uploaded successfully"


class PipelineRequest(BaseModel):
    experiment_id: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    selected_models: Optional[list[str]] = None
    async_mode: bool = Field(default=True, description="Run pipeline asynchronously via Celery")


class PipelineStatusResponse(BaseModel):
    experiment_id: str
    task_id: Optional[str] = None
    status: str  # pending | running | completed | failed
    current_stage: Optional[str] = None
    current_agent: Optional[str] = None
    agent_executions: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = []
    errors: list[str] = []
    progress_pct: float = 0.0


class PipelineResultResponse(BaseModel):
    experiment_id: str
    status: str
    best_model_name: Optional[str] = None
    leaderboard: list[dict[str, Any]] = []
    eda_insights: list[str] = []
    feature_importance: dict[str, float] = {}
    pdf_report_path: Optional[str] = None
    markdown_report_path: Optional[str] = None
    eda_charts: list[str] = []
    similar_experiments: list[dict[str, Any]] = []
    confidence_score: float = 0.0
    pipeline_complete: bool = False


class ExperimentListItem(BaseModel):
    experiment_id: str
    filename: Optional[str] = None
    task_type: Optional[str] = None
    best_model: Optional[str] = None
    timestamp: Optional[str] = None


class ExperimentDetailResponse(BaseModel):
    experiment_id: str
    metadata: dict[str, Any]
    document: str


class SimilarExperimentsRequest(BaseModel):
    query: str
    n_results: int = Field(default=5, ge=1, le=20)


class ChatMessage(BaseModel):
    experiment_id: str
    message: str


class ChatResponse(BaseModel):
    experiment_id: str
    response: str
    context_used: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
