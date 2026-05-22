"""
FastAPI Application Entry Point
================================
Production-grade FastAPI app with:
- CORS middleware
- Lifespan startup/shutdown
- Exception handlers
- Health check
- Celery task queue integration
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import upload, pipeline, experiments, reports
from app.utils.config import get_settings
from app.utils.logger import get_logger, setup_logging

logger = get_logger("api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    setup_logging()
    settings.ensure_directories()
    logger.info("api_startup", version=settings.app_version, env=settings.app_env.value)

    # Pre-warm LLM connection check
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                logger.info("ollama_available", models=models)
            else:
                logger.warning("ollama_unreachable", status=resp.status_code)
    except Exception as e:
        logger.warning("ollama_check_failed", error=str(e), fallback="huggingface")

    yield

    logger.info("api_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AutonomDS API",
        description="Autonomous Data Science Research Agent — REST API",
        version=settings.app_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ──────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response

    # ── Exception handlers ────────────────────────────────────────────────
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning("http_error", status=exc.status_code, detail=exc.detail, path=str(request.url))
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", error=str(exc), path=str(request.url), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc) if settings.debug else ""},
        )

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
    app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
    app.include_router(experiments.router, prefix="/api/v1", tags=["Experiments"])
    app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])

    # ── Health check ──────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "version": settings.app_version,
            "env": settings.app_env.value,
            "llm_provider": settings.llm_provider.value,
        }

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        return {"message": "AutonomDS API", "docs": "/docs"}

    return app


app = create_app()

# ── Celery App ────────────────────────────────────────────────────────────────
from celery import Celery

celery_app = Celery(
    "autonomds",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "app.api.main.run_pipeline_task": {"queue": "pipeline"},
    },
)


@celery_app.task(bind=True, name="app.api.main.run_pipeline_task", max_retries=2)
def run_pipeline_task(self, state: dict[str, Any]) -> dict[str, Any]:
    """Celery task: run the full LangGraph pipeline asynchronously."""
    from app.orchestration.graph import pipeline_graph
    from app.orchestration.state import AgentState

    try:
        config = {"configurable": {"thread_id": state.get("experiment_id", "default")}}
        result = pipeline_graph.invoke(state, config=config)
        return dict(result)
    except Exception as exc:
        logger.error("pipeline_task_failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=30)


def run() -> None:
    """Entrypoint for CLI."""
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if settings.is_production else 1,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
