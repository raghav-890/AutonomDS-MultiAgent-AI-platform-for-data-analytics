"""
Integration Tests — FastAPI Endpoints
=======================================
Tests the full API contract using httpx's TestClient.
No Celery, no Ollama required — uses sync pipeline mode.

Key notes:
- The upload route uses a module-level `settings = get_settings()` call,
  so we must override the upload_dir AFTER the app is created, not via patch.
- We test with a real temp directory for uploads.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """Create a TestClient for the FastAPI app with a real temp upload dir."""
    upload_dir = tmp_path_factory.mktemp("api_uploads")
    reports_dir = tmp_path_factory.mktemp("api_reports")

    # Patch get_settings before importing the app to set up dirs
    mock_cfg = MagicMock()
    mock_cfg.app_version = "1.0.0-test"
    mock_cfg.app_env.value = "test"
    mock_cfg.llm_provider.value = "ollama"
    mock_cfg.allowed_origins = ["*"]
    mock_cfg.is_production = False
    mock_cfg.is_development = True
    mock_cfg.debug = True
    mock_cfg.api_host = "0.0.0.0"
    mock_cfg.api_port = 8000
    mock_cfg.api_workers = 1
    mock_cfg.log_level = "INFO"
    mock_cfg.upload_dir = upload_dir
    mock_cfg.reports_dir = reports_dir
    mock_cfg.ensure_directories = MagicMock()
    mock_cfg.ollama_base_url = "http://localhost:11434"
    mock_cfg.chroma_persist_dir = tmp_path_factory.mktemp("chroma")
    mock_cfg.chroma_collection_name = "test_experiments"
    mock_cfg.max_upload_mb = 100
    mock_cfg.allowed_extensions = {".csv", ".xlsx", ".parquet", ".db", ".sqlite"}
    mock_cfg.mlflow_tracking_uri = f"file://{tmp_path_factory.mktemp('mlruns')}"
    mock_cfg.mlflow_experiment_name = "test"

    with patch("app.utils.config.get_settings", return_value=mock_cfg), \
         patch("app.api.routes.upload.get_settings", return_value=mock_cfg), \
         patch("app.api.routes.upload.settings", mock_cfg), \
         patch("app.api.routes.upload.validator") as mock_validator:

        # Configure the file validator mock
        import pandas as pd
        mock_validator.validate.return_value = (".csv", pd.DataFrame({"a": [1], "b": [2]}))
        mock_validator.safe_filename.side_effect = lambda f: f
        mock_validator.compute_checksum = MagicMock(return_value="abc123")

        from fastapi.testclient import TestClient
        from app.api.main import create_app
        app = create_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_root_endpoint(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "AutonomDS" in str(data)


# ── Upload endpoint ───────────────────────────────────────────────────────────

def test_upload_csv(api_client, tmp_path):
    """POST /api/v1/upload should accept a CSV and return 201 with experiment_id."""
    csv_bytes = b"age,income,target\n25,50000,0\n30,60000,1\n35,70000,0\n"
    files = {"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = api_client.post("/api/v1/upload", files=files)
    # With mocked validator, this should be 200 or 201
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert "experiment_id" in data
    assert len(data["experiment_id"]) > 5


# ── Pipeline status endpoint ──────────────────────────────────────────────────

def test_pipeline_status_not_found(api_client):
    """Status for unknown experiment should return a 'not_found' status (not raise 500)."""
    resp = api_client.get("/api/v1/pipeline/status/nonexistent-exp-id-xyz-999")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") in ("not_found", "PENDING", "unknown")


# ── Reports endpoint ──────────────────────────────────────────────────────────

def test_report_not_found(api_client):
    """Report for unknown experiment should return 404."""
    resp = api_client.get("/api/v1/reports/nonexistent-exp-12345/pdf")
    assert resp.status_code == 404


# ── OpenAPI docs ──────────────────────────────────────────────────────────────

def test_openapi_docs_accessible(api_client):
    """OpenAPI docs should be accessible."""
    resp = api_client.get("/docs")
    assert resp.status_code == 200


def test_openapi_json_accessible(api_client):
    """OpenAPI JSON schema should be accessible."""
    resp = api_client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "openapi" in schema
    assert "paths" in schema
