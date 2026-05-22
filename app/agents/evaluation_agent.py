"""
Evaluation Agent
=================
Computes full model evaluation suite:
- Classification: accuracy, F1, precision, recall, ROC-AUC, confusion matrix
- Regression: RMSE, MAE, R², residual analysis
- Builds ranked leaderboard
- Generates evaluation charts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.config import get_settings
from app.utils.helpers import now_iso

COLORS = {"primary": "#6366f1", "success": "#10b981", "danger": "#ef4444", "bg": "#0f172a", "card": "#1e293b", "text": "#e2e8f0"}


class EvaluationAgent(BaseAgent):
    name = "evaluation_agent"
    description = "Evaluates trained models and builds a performance leaderboard"
    stage = PipelineStage.EVALUATION
    max_retries = 1

    def execute(self, state: AgentState) -> AgentState:
        processed_path = state.get("processed_data_path")
        trained_models = state.get("trained_models", [])

        if not processed_path or not Path(processed_path).exists():
            raise AgentError("No processed data found.")
        if not trained_models:
            raise AgentError("No trained models found.")

        df = pd.read_parquet(processed_path)
        target_col = state.get("target_column", "")
        task_type = state.get("task_type", "regression")
        settings = get_settings()

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        is_classification = "classification" in task_type

        # Use fixed test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if is_classification else None
        )

        chart_dir = Path(settings.reports_dir) / state.get("experiment_id", "default") / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)

        leaderboard: list[dict[str, Any]] = []
        all_eval: dict[str, Any] = {}
        cm_data: list[list[int]] = []
        roc_auc = 0.0

        for i, model_info in enumerate(trained_models):
            if "error" in model_info:
                continue
            model_path = model_info.get("model_path")
            if not model_path or not Path(model_path).exists():
                continue

            try:
                model = joblib.load(model_path)
                model_name = model_info["model_name"]
                metrics: dict[str, float] = {}

                if is_classification:
                    y_pred = model.predict(X_test)
                    y_prob = None
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)
                    avg = "binary" if "binary" in task_type else "macro"
                    metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
                    metrics["f1"] = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)
                    metrics["precision"] = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
                    metrics["recall"] = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
                    if y_prob is not None:
                        try:
                            if "binary" in task_type:
                                metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
                            else:
                                metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"), 4)
                        except Exception:
                            pass
                    # Save confusion matrix for best model
                    if i == 0 or metrics.get("f1", 0) > leaderboard[-1].get("metrics", {}).get("f1", 0) if leaderboard else True:
                        cm_data = confusion_matrix(y_test, y_pred).tolist()
                    roc_auc = metrics.get("roc_auc", 0.0)
                else:
                    y_pred = model.predict(X_test)
                    metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                    metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
                    metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)

                entry = {
                    "rank": 0,
                    "model_name": model_name,
                    "metrics": metrics,
                    "training_time_seconds": model_info.get("training_time_seconds", 0),
                    "cv_mean": model_info.get("cv_mean", 0),
                }
                leaderboard.append(entry)
                all_eval[model_name] = metrics
            except Exception as e:
                self.logger.warning("eval_failed", model=model_info.get("model_name"), error=str(e))

        # Rank by primary metric
        primary = "roc_auc" if "binary" in task_type else ("accuracy" if is_classification else "r2")
        leaderboard.sort(key=lambda x: x["metrics"].get(primary, 0), reverse=True)
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        # Generate charts
        self._plot_leaderboard(leaderboard, primary, chart_dir)
        if cm_data and is_classification:
            self._plot_confusion_matrix(cm_data, chart_dir)

        eval_results = {
            "all_metrics": all_eval,
            "primary_metric": primary,
            "n_models_evaluated": len(leaderboard),
            "completed_at": now_iso(),
        }

        state = dict(state)  # type: ignore[assignment]
        state["evaluation_results"] = eval_results
        state["leaderboard"] = leaderboard
        state["confusion_matrix"] = cm_data
        state["roc_auc"] = roc_auc

        if leaderboard:
            state["best_model_name"] = leaderboard[0]["model_name"]
            best_path = next(
                (m.get("model_path") for m in trained_models if m.get("model_name") == leaderboard[0]["model_name"]),
                state.get("best_model_path", ""),
            )
            state["best_model_path"] = best_path or state.get("best_model_path", "")

        self.logger.info("evaluation_complete", n_models=len(leaderboard))
        return state  # type: ignore[return-value]

    def _plot_leaderboard(self, leaderboard: list[dict], metric: str, chart_dir: Path) -> None:
        try:
            names = [e["model_name"] for e in leaderboard]
            scores = [e["metrics"].get(metric, 0) for e in leaderboard]
            fig = go.Figure(go.Bar(
                x=scores, y=names, orientation="h",
                marker_color=[COLORS["primary"] if i == 0 else COLORS["card"] for i in range(len(names))],
                text=[f"{s:.4f}" for s in scores], textposition="outside",
            ))
            fig.update_layout(
                title=f"Model Leaderboard — {metric.upper()}",
                xaxis_title=metric, paper_bgcolor=COLORS["bg"],
                plot_bgcolor=COLORS["card"], font_color=COLORS["text"],
                height=max(300, len(names) * 60),
            )
            fig.write_html(str(chart_dir / "leaderboard.html"))
        except Exception as e:
            self.logger.warning("leaderboard_plot_failed", error=str(e))

    def _plot_confusion_matrix(self, cm: list[list[int]], chart_dir: Path) -> None:
        try:
            fig = go.Figure(go.Heatmap(
                z=cm, colorscale="Blues", showscale=True,
                text=cm, texttemplate="%{text}",
            ))
            fig.update_layout(
                title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual",
                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"], font_color=COLORS["text"],
            )
            fig.write_html(str(chart_dir / "confusion_matrix.html"))
        except Exception as e:
            self.logger.warning("cm_plot_failed", error=str(e))

    def compute_confidence(self, state: AgentState) -> float:
        lb = state.get("leaderboard", [])
        if not lb:
            return 0.0
        primary = state.get("evaluation_results", {}).get("primary_metric", "r2")
        best_score = lb[0]["metrics"].get(primary, 0)
        if "roc_auc" in primary or "accuracy" in primary:
            return min(1.0, best_score)
        return min(1.0, max(0.0, best_score))

    def _success_message(self, state: AgentState) -> str:
        lb = state.get("leaderboard", [])
        if lb:
            best = lb[0]
            primary = state.get("evaluation_results", {}).get("primary_metric", "score")
            score = best["metrics"].get(primary, 0)
            return f"✅ Evaluation complete. Best: **{best['model_name']}** ({primary}={score:.4f})"
        return "✅ Evaluation complete."
