"""
Explainability Agent
=====================
Generates SHAP + LIME explanations for the best model:
- SHAP TreeExplainer (for tree models) or KernelExplainer
- Feature importance plots
- LIME local explanations
- NL interpretability report via LLM
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.config import get_settings
from app.utils.helpers import now_iso

COLORS = {"primary": "#6366f1", "secondary": "#8b5cf6", "bg": "#0f172a", "card": "#1e293b", "text": "#e2e8f0"}


class ExplainabilityAgent(BaseAgent):
    name = "explainability_agent"
    description = "Generates SHAP and LIME model explanations"
    stage = PipelineStage.EXPLAINABILITY
    max_retries = 1

    def execute(self, state: AgentState) -> AgentState:
        best_model_path = state.get("best_model_path")
        processed_path = state.get("processed_data_path")

        if not best_model_path or not Path(best_model_path).exists():
            raise AgentError("Best model not found.")
        if not processed_path or not Path(processed_path).exists():
            raise AgentError("Processed data not found.")

        settings = get_settings()
        chart_dir = Path(settings.reports_dir) / state.get("experiment_id", "default") / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)

        model = joblib.load(best_model_path)
        df = pd.read_parquet(processed_path)
        target_col = state.get("target_column", "")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        feature_names = X.columns.tolist()

        # Sample for performance
        sample_size = min(200, len(X))
        X_sample = X.sample(sample_size, random_state=42)

        shap_chart_path = None
        feature_importance: dict[str, float] = {}

        # ── SHAP ─────────────────────────────────────────────────────────
        try:
            import shap
            model_type = type(model).__name__
            tree_based = any(t in model_type for t in ["Forest", "XGB", "LGBM", "Cat", "Tree", "Gradient"])

            if tree_based:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(
                    model.predict if hasattr(model, "predict_proba") else model.predict,
                    shap.sample(X_sample, 50),
                )

            shap_values = explainer.shap_values(X_sample)

            # Handle multi-output (classification)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values

            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            if len(mean_abs_shap) == len(feature_names):
                feature_importance = {
                    name: round(float(val), 4)
                    for name, val in zip(feature_names, mean_abs_shap)
                }
                feature_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
                shap_chart_path = self._plot_shap_importance(
                    feature_importance, chart_dir
                )

        except Exception as e:
            self.logger.warning("shap_failed", error=str(e))
            # Fallback: use sklearn feature_importances_ or coef_
            feature_importance = self._get_fallback_importance(model, feature_names)

        # ── LIME local explanation for first sample ────────────────────
        lime_explanation = ""
        try:
            from lime.lime_tabular import LimeTabularExplainer
            task_type = state.get("task_type", "regression")
            mode = "classification" if "classification" in task_type else "regression"
            lime_exp = LimeTabularExplainer(
                X.values,
                feature_names=feature_names,
                mode=mode,
                random_state=42,
            )
            predict_fn = model.predict_proba if (mode == "classification" and hasattr(model, "predict_proba")) else model.predict
            explanation = lime_exp.explain_instance(
                X_sample.iloc[0].values, predict_fn, num_features=10
            )
            lime_explanation = str(explanation.as_list())
        except Exception as e:
            self.logger.warning("lime_failed", error=str(e))

        # ── LLM interpretability report ───────────────────────────────
        report = self._generate_report(feature_importance, state)

        # Save SHAP values path reference
        shap_path = str(chart_dir / "shap_importance.html") if shap_chart_path else ""

        state = dict(state)  # type: ignore[assignment]
        state["shap_values_path"] = shap_path
        state["feature_importance"] = feature_importance
        state["explainability_report"] = report

        if shap_chart_path:
            charts = list(state.get("eda_charts", []))
            charts.append(shap_chart_path)
            state["eda_charts"] = charts

        self.logger.info("explainability_complete", top_features=list(feature_importance.keys())[:5])
        return state  # type: ignore[return-value]

    def _get_fallback_importance(self, model: Any, feature_names: list[str]) -> dict[str, float]:
        """Fallback importance from model attributes."""
        try:
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
                return dict(sorted(
                    {n: round(float(v), 4) for n, v in zip(feature_names, imps)}.items(),
                    key=lambda x: x[1], reverse=True
                ))
            if hasattr(model, "coef_"):
                coef = np.abs(model.coef_).flatten()
                if len(coef) == len(feature_names):
                    return dict(sorted(
                        {n: round(float(v), 4) for n, v in zip(feature_names, coef)}.items(),
                        key=lambda x: x[1], reverse=True
                    ))
        except Exception:
            pass
        return {}

    def _plot_shap_importance(self, importance: dict[str, float], chart_dir: Path) -> str:
        """Plot top-N SHAP feature importances."""
        top = list(importance.items())[:20]
        names = [t[0] for t in reversed(top)]
        values = [t[1] for t in reversed(top)]
        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
                showscale=False,
            ),
        ))
        fig.update_layout(
            title="SHAP Feature Importance (Mean |SHAP|)",
            xaxis_title="Mean |SHAP Value|",
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
            font_color=COLORS["text"], height=max(400, len(top) * 28),
        )
        path = str(chart_dir / "shap_importance.html")
        fig.write_html(path)
        return path

    def _generate_report(self, importance: dict[str, float], state: AgentState) -> str:
        """Generate NL explainability report via LLM."""
        top_features = list(importance.keys())[:10]
        system = (
            "You are an expert ML explainability engineer. "
            "Generate a concise model interpretability report (3-5 paragraphs) covering: "
            "1) Most important features and why they matter, "
            "2) Model behavior insights, "
            "3) Business recommendations based on feature importance."
        )
        user = (
            f"Model: {state.get('best_model_name', 'Unknown')}\n"
            f"Task: {state.get('task_type', 'unknown')}\n"
            f"Top features by SHAP: {top_features}\n"
            f"Leaderboard score: {state.get('leaderboard', [{}])[0].get('metrics', {}) if state.get('leaderboard') else {}}\n"
            "Write a model interpretability report:"
        )
        return self.ask_llm(system, user)

    def _success_message(self, state: AgentState) -> str:
        n_features = len(state.get("feature_importance", {}))
        return f"✅ Explainability: SHAP importance computed for {n_features} features."
