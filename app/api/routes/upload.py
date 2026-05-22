"""
Upload Route
=============
Handles file uploads with validation, checksum computation,
and initial dataset profiling.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.schemas import UploadResponse
from app.utils.config import get_settings
from app.utils.helpers import generate_experiment_id, now_iso
from app.utils.logger import get_logger
from app.utils.validators import FileValidator, ValidationError

router = APIRouter()
logger = get_logger("upload_route")
settings = get_settings()
validator = FileValidator()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a dataset file",
)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file (CSV, Excel, Parquet, SQLite)"),
    target_column: str | None = Form(default=None, description="Target column name (optional)"),
) -> UploadResponse:
    """
    Upload a dataset file and receive an experiment_id for pipeline execution.

    Supported formats: CSV, XLSX, XLS, Parquet, SQLite (.db, .sqlite)
    Max size: configurable via MAX_UPLOAD_SIZE_MB env var (default 200MB)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    logger.info("upload_received", filename=file.filename, content_type=file.content_type)

    # Read file content
    content = await file.read()

    # Validate
    try:
        ext, df = validator.validate(file.filename, content)
    except ValidationError as e:
        logger.warning("upload_validation_failed", filename=file.filename, error=e.message)
        raise HTTPException(status_code=422, detail=e.message)

    # Generate experiment ID and save file
    exp_id = generate_experiment_id()
    safe_name = validator.safe_filename(file.filename)
    upload_path = settings.upload_dir / exp_id
    upload_path.mkdir(parents=True, exist_ok=True)
    file_path = upload_path / safe_name

    with open(file_path, "wb") as f:
        f.write(content)

    checksum = FileValidator.compute_checksum(content)

    # Quick stats from in-memory DataFrame (if available)
    n_rows = len(df) if df is not None else 0
    n_cols = len(df.columns) if df is not None else 0
    columns = df.columns.tolist() if df is not None else []

    # Detect target
    from app.utils.helpers import detect_target_column, infer_task_type
    detected_target = target_column
    task_type_str = "unknown"
    if df is not None:
        if not detected_target:
            detected_target = detect_target_column(df)
        if detected_target and detected_target in df.columns:
            task_type_str = infer_task_type(df, detected_target)

    logger.info(
        "upload_success",
        exp_id=exp_id,
        filename=safe_name,
        rows=n_rows,
        cols=n_cols,
    )

    return UploadResponse(
        experiment_id=exp_id,
        filename=safe_name,
        checksum=checksum,
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        detected_target=detected_target,
        task_type=task_type_str,
        file_path=str(file_path),
    )
