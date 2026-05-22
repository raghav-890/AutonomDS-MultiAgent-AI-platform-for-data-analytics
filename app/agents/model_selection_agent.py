"""
Model Selection Agent
======================
Determines task type and selects best candidate models.
Runs fast baseline benchmarking to rank initial candidates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.helpers import now_iso


# Model catalog by task type
MODEL_CATALOG: dict[str, list[dict[str, Any]]] = {
    "binary_classification": [
        {"name": "LogisticRegression", "type": "linear", "fast": True},
        {"name": "RandomForestClassifier", "type": "ensemble", "fast": True},
        {"name": "XGBClassifier", "type": "boosting", "fast": True},
        {"name": "LGBMClassifier", "type": "boosting", "fast": True},
        {"name": "CatBoostClassifier", "type": "boosting", "fast": False},
        {"name": "SVC", "type": "kernel", "fast": False},
        {"name": "MLPClassifier", "type": "neural", "fast": False},
    ],
    "multiclass_classification": [
        {"name": "LogisticRegression", "type": "linear", "fast": True},
        {"name": "RandomForestClassifier", "type": "ensemble", "fast": True},
        {"name": "XGBClassifier", "type": "boosting", "fast": True},
        {"name": "LGBMClassifier", "type": "boosting", "fast": True},
        {"name": "CatBoostClassifier", "type": "boosting", "fast": False},
    ],
    "regression": [
        {"name": "Ridge", "type": "linear", "fast": True},
        {"name": "RandomForestRegressor", "type": "ensemble", "fast": True},
        {"name": "XGBRegressor", "type": "boosting", "fast": True},
        {"name": "LGBMRegressor", "type": "boosting", "fast": True},
        {"name": "CatBoostRegressor", "type": "boosting", "fast": False},
        {"name": "SVR", "type": "kernel", "fast": False},
    ],
}


class ModelSelectionAgent(BaseAgent):
    name = "model_selection_agent"
    description = "Selects optimal ML models based on task type and data characteristics"
    stage = PipelineStage.MODEL_SELECTION
    max_retries = 1

    def execute(self, state: AgentState) -> AgentState:
        processed_path = state.get("processed_data_path")
        if not processed_path or not Path(processed_path).exists():
            raise AgentError("No feature-engineered data found.")

        df = pd.read_parquet(processed_path)
        target_col = state.get("target_column", "")
        task_type = state.get("task_type", "regression")

        if not target_col or target_col not in df.columns:
            raise AgentError(f"Target column '{target_col}' not found.")

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        # Normalize task_type to catalog key
        catalog_key = self._resolve_catalog_key(task_type)
        candidates = MODEL_CATALOG.get(catalog_key, MODEL_CATALOG["regression"])

        # Filter by dataset size (skip slow models on large data for speed)
        n_rows = len(df)
        if n_rows > 20000:
            candidates = [m for m in candidates if m["fast"]]

        # Run baseline quick evaluation
        baseline_results = self._run_baselines(X, y, task_type, catalog_key)

        # Ask LLM to recommend best models given context
        recommended = self._llm_recommend(candidates, state, n_rows)

        state = dict(state)  # type: ignore[assignment]
        state["candidate_models"] = [m["name"] for m in candidates]
        state["selected_model_types"] = recommended
        state["baseline_results"] = baseline_results

        self.logger.info(
            "model_selection_complete",
            candidates=len(candidates),
            selected=len(recommended),
        )
        return state  # type: ignore[return-value]

    def _resolve_catalog_key(self, task_type: str) -> str:
        if "binary" in task_type:
            return "binary_classification"
        if "multiclass" in task_type:
            return "multiclass_classification"
        return "regression"

    def _run_baselines(
        self, X: np.ndarray, y: np.ndarray, task_type: str, catalog_key: str
    ) -> dict[str, Any]:
        """Run a fast DecisionTree + Dummy baseline for comparison."""
        results: dict[str, Any] = {}
        cv = 3
        scoring = "roc_auc" if "binary" in task_type else ("accuracy" if "classification" in task_type else "r2")

        baselines: list[Any] = []
        if "classification" in task_type:
            baselines = [
                ("DummyClassifier", DummyClassifier(strategy="most_frequent")),
                ("DecisionTree", DecisionTreeClassifier(max_depth=3, random_state=42)),
            ]
        else:
            baselines = [
                ("DummyRegressor", DummyRegressor()),
                ("DecisionTree", DecisionTreeRegressor(max_depth=3, random_state=42)),
            ]

        for name, model in baselines:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                results[name] = {
                    "mean": round(float(scores.mean()), 4),
                    "std": round(float(scores.std()), 4),
                    "metric": scoring,
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        return results

    def _llm_recommend(
        self,
        candidates: list[dict[str, Any]],
        state: AgentState,
        n_rows: int,
    ) -> list[str]:
        """Ask LLM to pick best 3-4 models from candidates."""
        candidate_names = [m["name"] for m in candidates]
        system = (
            "You are an expert ML engineer. Given dataset characteristics and candidate models, "
            "select the best 3-4 models to train. Return ONLY a JSON list of model names, "
            "e.g. [\"XGBClassifier\", \"RandomForestClassifier\", \"LogisticRegression\"]"
        )
        user = (
            f"Task: {state.get('task_type')}\n"
            f"Dataset: {n_rows:,} rows × {state.get('n_features_selected', '?')} features\n"
            f"Candidates: {candidate_names}\n"
            f"Missing: {state.get('dataset_info', {}).get('missing_pct', 0):.1f}%\n"
            "Select 3-4 best models (consider data size, task, and diversity)."
        )
        response = self.ask_llm(system, user)
        # Parse JSON list from response
        import re, json
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                selected = json.loads(match.group())
                # Validate names are in candidates
                valid = [n for n in selected if n in candidate_names]
                if valid:
                    return valid
            except Exception:
                pass
        # Fallback: pick first 3 fast models
        return [m["name"] for m in candidates if m.get("fast")][:3]

    def _success_message(self, state: AgentState) -> str:
        return (
            f"✅ Model selection: {len(state.get('selected_model_types', []))} models selected: "
            f"{', '.join(state.get('selected_model_types', []))}"
        )
