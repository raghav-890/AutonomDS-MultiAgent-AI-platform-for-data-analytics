"""
Data Cleaning Agent
====================
Handles all data preprocessing:
- Missing value imputation (smart strategy per column)
- Categorical encoding (label, ordinal, one-hot, target encoding)
- Scaling & normalization
- Duplicate removal
- Leakage detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
)

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.helpers import now_iso


class DataCleaningAgent(BaseAgent):
    name = "data_cleaning_agent"
    description = "Cleans data: imputation, encoding, scaling, deduplication"
    stage = PipelineStage.CLEANING
    max_retries = 1

    def execute(self, state: AgentState) -> AgentState:
        processed_path = state.get("processed_data_path")
        if not processed_path or not Path(processed_path).exists():
            raise AgentError("No processed data found. Run ingestion first.")

        df = pd.read_parquet(processed_path)
        target_col = state.get("target_column", "")
        task_type = state.get("task_type", "regression")
        numeric_cols = state.get("numeric_columns", [])
        categorical_cols = state.get("categorical_columns", [])
        actions: list[str] = []

        # 1. Remove duplicates
        n_dups = df.duplicated().sum()
        if n_dups > 0:
            df = df.drop_duplicates()
            actions.append(f"Removed {n_dups} duplicate rows")

        # 2. Drop columns with >60% missing
        missing_pct = df.isnull().mean()
        drop_cols = missing_pct[missing_pct > 0.6].index.tolist()
        drop_cols = [c for c in drop_cols if c != target_col]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            actions.append(f"Dropped high-missing columns: {drop_cols}")
            numeric_cols = [c for c in numeric_cols if c not in drop_cols]
            categorical_cols = [c for c in categorical_cols if c not in drop_cols]

        # 3. Impute numeric columns
        num_missing = [c for c in numeric_cols if c in df.columns and df[c].isnull().any() and c != target_col]
        if num_missing:
            if len(df) < 5000:
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy="median")
            df[num_missing] = imputer.fit_transform(df[num_missing])
            actions.append(f"Imputed {len(num_missing)} numeric columns")

        # 4. Impute categorical columns
        cat_missing = [c for c in categorical_cols if c in df.columns and df[c].isnull().any() and c != target_col]
        if cat_missing:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[cat_missing] = cat_imputer.fit_transform(df[cat_missing])
            actions.append(f"Imputed {len(cat_missing)} categorical columns")

        # 5. Encode categoricals
        encoder_map: dict[str, str] = {}
        for col in categorical_cols:
            if col not in df.columns or col == target_col:
                continue
            n_unique = df[col].nunique()
            if n_unique <= 2:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoder_map[col] = "label"
            elif n_unique <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                encoder_map[col] = "onehot"
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoder_map[col] = "label_high_cardinality"

        actions.append(f"Encoded {len(encoder_map)} categorical columns")

        # 6. Encode target if classification
        if target_col and target_col in df.columns:
            if "classification" in task_type:
                if not pd.api.types.is_numeric_dtype(df[target_col]):
                    le = LabelEncoder()
                    df[target_col] = le.fit_transform(df[target_col].astype(str))
                    actions.append("Encoded target column")

        # 7. Scale numeric features
        feature_num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col and c in df.columns
        ]
        if feature_num_cols:
            scaler = RobustScaler()
            df[feature_num_cols] = scaler.fit_transform(df[feature_num_cols])
            actions.append(f"Scaled {len(feature_num_cols)} numeric features (RobustScaler)")

        # 8. Leakage detection (features with correlation > 0.99 to target)
        leakage_cols: list[str] = []
        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            num_feats = [c for c in feature_num_cols if c in df.columns]
            for col in num_feats:
                try:
                    corr = abs(df[col].corr(df[target_col]))
                    if corr > 0.99:
                        leakage_cols.append(col)
                except Exception:
                    pass
            if leakage_cols:
                df = df.drop(columns=leakage_cols)
                actions.append(f"Removed potential leakage columns: {leakage_cols}")

        # Save cleaned data
        settings_upload = Path(processed_path).parent
        cleaned_path = settings_upload / f"{Path(processed_path).stem}_cleaned.parquet"
        df.to_parquet(cleaned_path, index=False)

        cleaning_report = {
            "actions": actions,
            "n_rows_after": len(df),
            "n_cols_after": len(df.columns),
            "encoder_map": encoder_map,
            "leakage_cols_removed": leakage_cols,
            "completed_at": now_iso(),
        }

        state = dict(state)  # type: ignore[assignment]
        state["processed_data_path"] = str(cleaned_path)
        state["cleaning_report"] = cleaning_report
        state["cleaning_actions"] = actions
        state["numeric_columns"] = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
        state["categorical_columns"] = []

        self.logger.info("cleaning_complete", actions=len(actions), rows=len(df))
        return state  # type: ignore[return-value]

    def _success_message(self, state: AgentState) -> str:
        report = state.get("cleaning_report", {})
        return (
            f"✅ Data cleaning complete: {len(report.get('actions', []))} actions. "
            f"Output: {report.get('n_rows_after', 0):,} rows × {report.get('n_cols_after', 0)} cols."
        )
