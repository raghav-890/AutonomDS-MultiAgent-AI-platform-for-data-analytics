"""
MLflow Experiment Tracker
==========================
Local MLflow integration for experiment tracking.
All data stored locally — no cloud account needed.

Features:
- Experiment/run lifecycle management
- Parameter and metric logging
- Model artifact tracking
- Run history retrieval for the Experiments page
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Generator

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("mlflow_tracker")


class MLflowTracker:
    """
    Thin wrapper around MLflow local tracking.

    Degrades gracefully if MLflow is not installed or not configured —
    all methods become no-ops so the pipeline continues without tracking.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._available = False
        self._mlflow: Any = None
        self._init()

    def _init(self) -> None:
        """Attempt to initialise MLflow."""
        try:
            import mlflow
            tracking_uri = str(
                Path(self.settings.reports_dir).parent / "mlflow" / "mlruns"
            )
            Path(tracking_uri).mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{tracking_uri}")
            self._mlflow = mlflow
            self._available = True
            logger.info("mlflow_initialised", tracking_uri=tracking_uri)
        except ImportError:
            logger.warning("mlflow_not_installed")
        except Exception as e:
            logger.warning("mlflow_init_failed", error=str(e))

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Experiment management ─────────────────────────────────────────────────

    def get_or_create_experiment(self, name: str) -> str | None:
        """Get an existing MLflow experiment or create one."""
        if not self._available:
            return None
        try:
            exp = self._mlflow.get_experiment_by_name(name)
            if exp is None:
                exp_id = self._mlflow.create_experiment(name)
                logger.info("mlflow_experiment_created", name=name, exp_id=exp_id)
                return exp_id
            return exp.experiment_id
        except Exception as e:
            logger.warning("mlflow_experiment_error", error=str(e))
            return None

    @contextlib.contextmanager
    def start_run(
        self,
        experiment_name: str = "AutonomDS",
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[str | None, None, None]:
        """Context manager that starts an MLflow run and returns its ID."""
        if not self._available:
            yield None
            return

        exp_id = self.get_or_create_experiment(experiment_name)
        try:
            with self._mlflow.start_run(
                experiment_id=exp_id,
                run_name=run_name,
                tags=tags or {},
            ) as run:
                logger.info("mlflow_run_started", run_id=run.info.run_id)
                yield run.info.run_id
        except Exception as e:
            logger.warning("mlflow_run_failed", error=str(e))
            yield None

    # ── Logging primitives ────────────────────────────────────────────────────

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if not self._available:
            return
        try:
            # MLflow only accepts string values for params
            str_params = {k: str(v)[:500] for k, v in params.items()}
            self._mlflow.log_params(str_params)
        except Exception as e:
            logger.warning("mlflow_log_params_failed", error=str(e))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics."""
        if not self._available:
            return
        try:
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            self._mlflow.log_metrics(clean, step=step)
        except Exception as e:
            logger.warning("mlflow_log_metrics_failed", error=str(e))

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file artifact (model, chart, report)."""
        if not self._available:
            return
        try:
            path = Path(local_path)
            if path.exists():
                self._mlflow.log_artifact(str(path))
        except Exception as e:
            logger.warning("mlflow_log_artifact_failed", error=str(e))

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set run tags."""
        if not self._available:
            return
        try:
            self._mlflow.set_tags(tags)
        except Exception as e:
            logger.warning("mlflow_set_tags_failed", error=str(e))

    # ── Query past runs ───────────────────────────────────────────────────────

    def list_runs(self, experiment_name: str = "AutonomDS", max_results: int = 20) -> list[dict]:
        """Return recent MLflow runs as plain dicts."""
        if not self._available:
            return []
        try:
            exp = self._mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                return []
            runs = self._mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )
            if runs.empty:
                return []
            return runs.to_dict(orient="records")
        except Exception as e:
            logger.warning("mlflow_list_runs_failed", error=str(e))
            return []

    def log_pipeline_state(self, state: dict[str, Any]) -> None:
        """
        Convenience: log all relevant fields from an AgentState dict.
        Called by the pipeline after completion.
        """
        if not self._available:
            return
        try:
            # Params
            dataset_info = state.get("dataset_info", {})
            self.log_params({
                "experiment_id": state.get("experiment_id", ""),
                "task_type":     state.get("task_type", ""),
                "target_column": state.get("target_column", ""),
                "n_rows":        dataset_info.get("n_rows", 0),
                "n_cols":        dataset_info.get("n_cols", 0),
                "best_model":    state.get("best_model_name", ""),
                "n_features":    state.get("n_features_selected", 0),
                "retry_count":   state.get("retry_count", 0),
            })

            # Metrics from leaderboard
            lb = state.get("leaderboard", [])
            if lb:
                best_metrics = lb[0].get("metrics", {})
                self.log_metrics(best_metrics)

            # Tags
            self.set_tags({
                "pipeline": "autonomds",
                "status": "completed" if state.get("pipeline_complete") else "partial",
                "confidence": str(round(state.get("confidence_score", 0), 3)),
            })

        except Exception as e:
            logger.warning("mlflow_pipeline_log_failed", error=str(e))


# ── Singleton ─────────────────────────────────────────────────────────────────
_tracker: MLflowTracker | None = None


def get_tracker() -> MLflowTracker:
    """Return a cached MLflowTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker()
    return _tracker
