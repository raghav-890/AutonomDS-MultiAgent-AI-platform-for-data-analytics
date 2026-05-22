"""Experiments and Reports Routes."""
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.api.schemas import ExperimentListItem, ExperimentDetailResponse, SimilarExperimentsRequest
from app.utils.config import get_settings
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger("experiments_route")
settings = get_settings()


@router.get("/experiments", response_model=list[ExperimentListItem])
async def list_experiments() -> list[ExperimentListItem]:
    """List all experiments stored in memory."""
    try:
        from app.memory.experiment_memory import ExperimentMemory
        mem = ExperimentMemory()
        all_exps = mem.get_all_experiments()
        return [
            ExperimentListItem(
                experiment_id=e["id"],
                filename=e["metadata"].get("filename"),
                task_type=e["metadata"].get("task_type"),
                best_model=e["metadata"].get("best_model"),
                timestamp=e["metadata"].get("timestamp"),
            )
            for e in all_exps
        ]
    except Exception as e:
        logger.warning("list_experiments_failed", error=str(e))
        return []


@router.post("/experiments/similar", response_model=list[ExperimentDetailResponse])
async def find_similar(req: SimilarExperimentsRequest) -> list[ExperimentDetailResponse]:
    """Find experiments similar to a query."""
    try:
        from app.memory.experiment_memory import ExperimentMemory
        mem = ExperimentMemory()
        results = mem.find_similar(req.query, n_results=req.n_results)
        return [
            ExperimentDetailResponse(
                experiment_id=r["id"],
                metadata=r["metadata"],
                document=r["document"],
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
