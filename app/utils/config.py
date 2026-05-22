"""
AutonomDS Configuration System
================================
Centralized, typed configuration using Pydantic BaseSettings.
All values are loaded from environment variables / .env file.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Enumerations ─────────────────────────────────────────────────────────────

class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


class TrainingDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


# ── Settings Classes ──────────────────────────────────────────────────────────

class AppSettings(BaseSettings):
    """Core application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="AutonomDS", description="Application name")
    app_env: AppEnvironment = Field(default=AppEnvironment.DEVELOPMENT)
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    secret_key: str = Field(default="change-me-in-production")

    # API Server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=2)
    allowed_origins: list[str] = Field(default=["http://localhost:8501"])

    # Streamlit
    streamlit_port: int = Field(default=8501)
    streamlit_api_url: str = Field(default="http://localhost:8000")

    # LLM
    llm_provider: LLMProvider = Field(default=LLMProvider.AUTO)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral:7b-instruct")
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    ollama_timeout: int = Field(default=120)
    huggingface_api_token: Optional[str] = Field(default=None)
    huggingface_model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2")
    code_llm_model: str = Field(default="deepseek-coder:6.7b-instruct")

    # Embeddings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")

    # ChromaDB
    chroma_persist_dir: Path = Field(default=Path("./experiments/chroma_db"))
    chroma_collection_name: str = Field(default="experiment_memory")

    # MLflow
    mlflow_tracking_uri: str = Field(default="sqlite:///./experiments/mlflow.db")
    mlflow_experiment_name: str = Field(default="autonomds-experiments")
    mlflow_artifact_root: str = Field(default="./experiments/mlflow_artifacts")

    # Redis / Celery
    redis_url: str = Field(default="redis://localhost:6379/0")
    celery_broker_url: str = Field(default="redis://localhost:6379/0")
    celery_result_backend: str = Field(default="redis://localhost:6379/1")

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./experiments/autonomds.db")

    # Storage
    upload_dir: Path = Field(default=Path("./uploads"))
    reports_dir: Path = Field(default=Path("./experiments/reports"))
    max_upload_size_mb: int = Field(default=200, gt=0)
    allowed_extensions: list[str] = Field(
        default=["csv", "xlsx", "xls", "parquet", "sqlite", "db"]
    )

    # Kaggle
    kaggle_username: Optional[str] = Field(default=None)
    kaggle_key: Optional[str] = Field(default=None)

    # Training
    training_device: TrainingDevice = Field(default=TrainingDevice.AUTO)
    max_training_time_seconds: int = Field(default=300, gt=0)
    optuna_n_trials: int = Field(default=30, gt=0)
    cross_val_folds: int = Field(default=5, ge=2)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: LogFormat = Field(default=LogFormat.CONSOLE)
    log_file: Optional[Path] = Field(default=None)

    # Security
    sandbox_timeout_seconds: int = Field(default=30, gt=0)
    max_code_length: int = Field(default=10000, gt=0)
    enable_sandboxed_execution: bool = Field(default=True)

    # ── Validators ─────────────────────────────────────────────────────

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v

    # ── Computed Properties ────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnvironment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app_env == AppEnvironment.DEVELOPMENT

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def effective_llm_provider(self) -> LLMProvider:
        """Resolve AUTO provider: Ollama if reachable, else HuggingFace."""
        if self.llm_provider != LLMProvider.AUTO:
            return self.llm_provider
        # Check Ollama availability
        try:
            import httpx
            response = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=3.0)
            if response.status_code == 200:
                return LLMProvider.OLLAMA
        except Exception:
            pass
        return LLMProvider.HUGGINGFACE

    @property
    def effective_training_device(self) -> str:
        """Resolve AUTO device: CUDA > MPS > CPU."""
        if self.training_device != TrainingDevice.AUTO:
            return self.training_device.value
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ── Directory Setup ─────────────────────────────────────────────────

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        dirs = [
            self.upload_dir,
            self.reports_dir,
            self.chroma_persist_dir,
            Path("./experiments/mlflow_artifacts"),
            Path("./logs"),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings (singleton)."""
    settings = AppSettings()
    settings.ensure_directories()
    return settings


# ── Module-level convenience alias ────────────────────────────────────────────
settings = get_settings()
