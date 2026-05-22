"""
Feature Engineering Agent
===========================
Performs automated feature engineering:
- Feature selection (variance + correlation + RFECV)
- Polynomial / interaction features
- PCA dimensionality reduction
- Feature importance analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    VarianceThreshold, mutual_info_classif, mutual_info_regression,
)
from sklearn.preprocessing import PolynomialFeatures

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.helpers import now_iso


class FeatureEngineeringAgent(BaseAgent):
    name = "feature_engineering_agent"
    description = "Selects, creates, and transforms features for ML"
    stage = PipelineStage.FEATURE_ENGINEERING
    max_retries = 1

    def execute(self, state: AgentState) -> AgentState:
        processed_path = state.get("processed_data_path")
        if not processed_path or not Path(processed_path).exists():
            raise AgentError("No cleaned data found.")

        df = pd.read_parquet(processed_path)
        target_col = state.get("target_column", "")
        task_type = state.get("task_type", "regression")

        if not target_col or target_col not in df.columns:
            raise AgentError(f"Target column '{target_col}' not found in data.")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        n_orig = X.shape[1]

        feature_report: dict[str, Any] = {"n_features_original": n_orig}
        actions: list[str] = []

        # 1. Remove zero-variance features
        vt = VarianceThreshold(threshold=0.01)
        X_arr = vt.fit_transform(X)
        selected_mask = vt.get_support()
        X = X.loc[:, selected_mask]
        removed = n_orig - X.shape[1]
        if removed > 0:
            actions.append(f"Removed {removed} zero/low-variance features")

        # 2. Remove highly correlated features
        if X.shape[1] > 1:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
            if drop_corr:
                X = X.drop(columns=drop_corr)
                actions.append(f"Removed {len(drop_corr)} highly correlated features")

        # 3. Feature importance via RandomForest
        importance_dict: dict[str, float] = {}
        if X.shape[1] > 0:
            try:
                n_estimators = 50
                if "classification" in task_type:
                    rf = RandomForestClassifier(
                        n_estimators=n_estimators, max_depth=5, random_state=42, n_jobs=-1
                    )
                else:
                    rf = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=5, random_state=42, n_jobs=-1
                    )
                rf.fit(X, y)
                importance_dict = dict(zip(X.columns, rf.feature_importances_.round(4).tolist()))
                importance_dict = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
                # Keep top 80% cumulative importance features
                importances = np.array(list(importance_dict.values()))
                cumulative = np.cumsum(importances)
                threshold_idx = np.searchsorted(cumulative, 0.95) + 1
                top_features = list(importance_dict.keys())[:threshold_idx]
                X = X[top_features]
                actions.append(f"Selected top {len(top_features)} features by importance (95% cumulative)")
            except Exception as e:
                self.logger.warning("rf_importance_failed", error=str(e))

        # 4. Polynomial features (only for small feature sets)
        if 2 <= X.shape[1] <= 8 and len(df) < 50000:
            try:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_poly = poly.fit_transform(X)
                poly_names = poly.get_feature_names_out(X.columns)
                X = pd.DataFrame(X_poly, columns=poly_names, index=X.index)
                actions.append(f"Added {X.shape[1] - n_orig} interaction features")
            except Exception as e:
                self.logger.warning("poly_features_failed", error=str(e))

        # 5. PCA dimensionality reduction (if still too many features)
        pca_applied = False
        if X.shape[1] > 50:
            try:
                n_components = min(30, X.shape[1] - 1)
                pca = PCA(n_components=n_components, random_state=42)
                X_pca = pca.fit_transform(X)
                pca_cols = [f"PC_{i+1}" for i in range(n_components)]
                X = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
                var_explained = float(pca.explained_variance_ratio_.sum())
                actions.append(
                    f"PCA: reduced to {n_components} components "
                    f"({var_explained:.1%} variance explained)"
                )
                pca_applied = True
            except Exception as e:
                self.logger.warning("pca_failed", error=str(e))

        # Rebuild full DataFrame
        df_final = X.copy()
        df_final[target_col] = y.values

        settings_path = Path(processed_path).parent
        featured_path = settings_path / f"{Path(processed_path).stem}_featured.parquet"
        df_final.to_parquet(featured_path, index=False)

        feature_report.update({
            "actions": actions,
            "n_features_final": X.shape[1],
            "feature_importance": importance_dict,
            "pca_applied": pca_applied,
            "selected_features": X.columns.tolist(),
            "completed_at": now_iso(),
        })

        state = dict(state)  # type: ignore[assignment]
        state["processed_data_path"] = str(featured_path)
        state["feature_report"] = feature_report
        state["selected_features"] = X.columns.tolist()
        state["feature_importance"] = importance_dict
        state["n_features_original"] = n_orig
        state["n_features_selected"] = X.shape[1]

        self.logger.info("feature_engineering_complete", n_features=X.shape[1])
        return state  # type: ignore[return-value]

    def _success_message(self, state: AgentState) -> str:
        report = state.get("feature_report", {})
        return (
            f"✅ Feature engineering: {report.get('n_features_original', '?')} → "
            f"{report.get('n_features_final', '?')} features. "
            f"Actions: {len(report.get('actions', []))}."
        )
