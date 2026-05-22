"""Reports download route."""
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.utils.config import get_settings
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger("reports_route")
settings = get_settings()


@router.get("/reports/{experiment_id}/pdf", summary="Download PDF report")
async def download_pdf(experiment_id: str) -> FileResponse:
    path = settings.reports_dir / experiment_id / "report.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found.")
    return FileResponse(path, media_type="application/pdf", filename=f"report_{experiment_id}.pdf")


@router.get("/reports/{experiment_id}/markdown", summary="Download Markdown report")
async def download_markdown(experiment_id: str) -> FileResponse:
    path = settings.reports_dir / experiment_id / "report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Markdown report not found.")
    return FileResponse(path, media_type="text/markdown", filename=f"report_{experiment_id}.md")


@router.get("/reports/{experiment_id}/summary", summary="Get JSON experiment summary")
async def get_summary(experiment_id: str) -> dict:
    import json
    path = settings.reports_dir / experiment_id / "summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary not found.")
    return json.loads(path.read_text())
