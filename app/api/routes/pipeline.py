"""
Pipeline Route
===============
Triggers and polls the LangGraph pipeline execution.
Supports both async (Celery) and sync modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.api.schemas import (
    PipelineRequest, PipelineStatusResponse, PipelineResultResponse, ChatMessage, ChatResponse
)
from app.utils.config import get_settings
from app.utils.helpers import generate_run_id, now_iso
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger("pipeline_route")
settings = get_settings()

# In-memory task registry (use Redis in production)
_task_registry: dict[str, dict[str, Any]] = {}
_result_cache: dict[str, dict[str, Any]] = {}


@router.post(
    "/pipeline/run",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger the full ML pipeline",
)
async def run_pipeline(request: PipelineRequest) -> dict[str, Any]:
    """
    Trigger the full autonomous ML pipeline for a given experiment.
    Supports async (Celery) or sync execution.
    """
    exp_id = request.experiment_id
    upload_dir = settings.upload_dir / exp_id

    # Find uploaded file
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found. Upload a file first.")

    files = list(upload_dir.iterdir())
    if not files:
        raise HTTPException(status_code=404, detail="No file found for this experiment.")

    file_path = str(files[0])

    # Build initial state
    initial_state: dict[str, Any] = {
        "experiment_id": exp_id,
        "run_id": generate_run_id(),
        "created_at": now_iso(),
        "raw_data_path": file_path,
        "target_column": request.target_column or "",
        "task_type": request.task_type or "",
        "current_stage": "ingestion",
        "current_agent": "",
        "agent_executions": [],
        "messages": [],
        "errors": [],
        "retry_count": 0,
        "should_reflect": False,
        "reflection_notes": [],
        "confidence_score": 0.0,
        "pipeline_complete": False,
    }

    if request.selected_models:
        initial_state["selected_model_types"] = request.selected_models

    if request.async_mode:
        # Submit to Celery
        try:
            from app.api.main import run_pipeline_task
            task = run_pipeline_task.delay(initial_state)
            _task_registry[exp_id] = {
                "task_id": task.id,
                "status": "pending",
                "submitted_at": now_iso(),
            }
            logger.info("pipeline_task_submitted", exp_id=exp_id, task_id=task.id)
            return {"experiment_id": exp_id, "task_id": task.id, "status": "pending", "mode": "async"}
        except Exception as e:
            logger.warning("celery_unavailable_fallback_sync", error=str(e))
            # Fall through to sync mode

    # Sync mode (fallback or explicit)
    try:
        from app.orchestration.graph import pipeline_graph
        config = {"configurable": {"thread_id": exp_id}}
        result = pipeline_graph.invoke(initial_state, config=config)
        _result_cache[exp_id] = dict(result)
        logger.info("pipeline_sync_complete", exp_id=exp_id)
        return {"experiment_id": exp_id, "status": "completed", "mode": "sync"}
    except Exception as e:
        logger.error("pipeline_failed", exp_id=exp_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


@router.get(
    "/pipeline/status/{experiment_id}",
    response_model=PipelineStatusResponse,
    summary="Get pipeline execution status",
)
async def get_pipeline_status(experiment_id: str) -> PipelineStatusResponse:
    """Get the current status of a pipeline execution."""
    # Check Celery task
    task_info = _task_registry.get(experiment_id)
    if task_info:
        try:
            from celery.result import AsyncResult
            from app.api.main import celery_app
            task = AsyncResult(task_info["task_id"], app=celery_app)
            status = task.status.lower()
            if status == "success":
                _result_cache[experiment_id] = task.result
                status = "completed"
        except Exception:
            status = task_info.get("status", "unknown")
    elif experiment_id in _result_cache:
        status = "completed"
    else:
        status = "not_found"

    result = _result_cache.get(experiment_id, {})
    return PipelineStatusResponse(
        experiment_id=experiment_id,
        task_id=task_info.get("task_id") if task_info else None,
        status=status,
        current_stage=result.get("current_stage"),
        current_agent=result.get("current_agent"),
        agent_executions=result.get("agent_executions", []),
        messages=result.get("messages", []),
        errors=result.get("errors", []),
        progress_pct=_compute_progress(result),
    )


@router.get(
    "/pipeline/result/{experiment_id}",
    response_model=PipelineResultResponse,
    summary="Get full pipeline results",
)
async def get_pipeline_result(experiment_id: str) -> PipelineResultResponse:
    """Get the complete results of a finished pipeline."""
    result = _result_cache.get(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="No results found. Pipeline may not have completed.")

    return PipelineResultResponse(
        experiment_id=experiment_id,
        status="completed",
        best_model_name=result.get("best_model_name"),
        leaderboard=result.get("leaderboard", []),
        eda_insights=result.get("eda_insights", []),
        feature_importance=result.get("feature_importance", {}),
        pdf_report_path=result.get("pdf_report_path"),
        markdown_report_path=result.get("markdown_report_path"),
        eda_charts=result.get("eda_charts", []),
        similar_experiments=result.get("similar_experiments", []),
        confidence_score=result.get("confidence_score", 0.0),
        pipeline_complete=result.get("pipeline_complete", False),
    )


@router.post("/pipeline/chat", response_model=ChatResponse, summary="Chat with experiment assistant")
async def chat_with_experiment(msg: ChatMessage) -> ChatResponse:
    """RAG-based chat about an experiment's results."""
    result = _result_cache.get(msg.experiment_id, {})

    # Build context from experiment results
    context = ""
    context_used = False
    if result:
        leaderboard = result.get("leaderboard", [])
        insights = result.get("eda_insights", [])
        context = (
            f"Experiment: {msg.experiment_id}\n"
            f"Dataset: {result.get('dataset_info', {}).get('filename', 'unknown')}\n"
            f"Best model: {result.get('best_model_name', 'N/A')}\n"
            f"Leaderboard: {leaderboard[:3]}\n"
            f"EDA insights: {insights[:3]}\n"
        )
        context_used = True

    # Also check ChromaDB for similar experiments
    try:
        from app.memory.experiment_memory import ExperimentMemory
        mem = ExperimentMemory()
        rag = mem.rag_context(msg.message)
        context += f"\n{rag}"
    except Exception:
        pass

    from app.agents.base_agent import _build_llm
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _build_llm()
    messages = [
        SystemMessage(content=(
            "You are an expert data science assistant. Answer questions about ML experiments. "
            "Use the provided context. Be concise and specific."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {msg.message}"),
    ]
    response = llm.invoke(messages).content

    return ChatResponse(
        experiment_id=msg.experiment_id,
        response=response,
        context_used=context_used,
    )


def _compute_progress(state: dict[str, Any]) -> float:
    """Estimate pipeline progress as a percentage."""
    stages = ["ingestion", "eda", "cleaning", "feature_engineering",
              "model_selection", "training", "evaluation", "explainability",
              "report", "memory"]
    current = state.get("current_stage", "")
    try:
        idx = stages.index(current)
        return round((idx + 1) / len(stages) * 100, 1)
    except ValueError:
        if state.get("pipeline_complete"):
            return 100.0
        return 0.0
