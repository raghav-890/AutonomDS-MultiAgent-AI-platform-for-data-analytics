"""
Training Agent
===============
Trains selected ML models with:
- Cross-validation
- Optuna hyperparameter tuning
- GPU support (PyTorch / XGBoost / LightGBM)
- MLflow experiment tracking
- Early stopping
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.config import get_settings
from app.utils.helpers import now_iso

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_model(name: str, params: dict[str, Any], task_type: str, device: str) -> Any:
    """Instantiate a model by name with given params."""
    use_gpu = device == "cuda"
    
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params, max_iter=1000, random_state=42)
    elif name == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params, random_state=42)
    elif name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif name == "XGBClassifier":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, random_state=42, tree_method="gpu_hist" if use_gpu else "hist", verbosity=0)
    elif name == "XGBRegressor":
        from xgboost import XGBRegressor
        return XGBRegressor(**params, random_state=42, tree_method="gpu_hist" if use_gpu else "hist", verbosity=0)
    elif name == "LGBMClassifier":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, random_state=42, device="gpu" if use_gpu else "cpu", verbose=-1)
    elif name == "LGBMRegressor":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, random_state=42, device="gpu" if use_gpu else "cpu", verbose=-1)
    elif name in ("CatBoostClassifier", "CatBoostRegressor"):
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            cls = CatBoostClassifier if "Classifier" in name else CatBoostRegressor
            return cls(**params, random_seed=42, verbose=0, task_type="GPU" if use_gpu else "CPU")
        except ImportError:
            # Fallback to LightGBM
            from lightgbm import LGBMClassifier, LGBMRegressor
            cls = LGBMClassifier if "Classifier" in name else LGBMRegressor
            return cls(random_state=42, verbose=-1)
    else:
        raise AgentError(f"Unknown model: {name}")


def _get_param_space(name: str, task_type: str) -> dict[str, Any]:
    """Return Optuna hyperparameter search space per model."""
    if "LogisticRegression" in name:
        return {"C": ("float_log", 1e-3, 10.0), "solver": ("categorical", ["lbfgs", "saga"])}
    elif "Ridge" in name:
        return {"alpha": ("float_log", 1e-2, 100.0)}
    elif "RandomForest" in name:
        return {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 3, 15),
            "min_samples_split": ("int", 2, 10),
        }
    elif "XGB" in name:
        return {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 1e-3, 0.3),
            "subsample": ("float", 0.6, 1.0),
        }
    elif "LGBM" in name:
        return {
            "n_estimators": ("int", 50, 300),
            "num_leaves": ("int", 20, 150),
            "learning_rate": ("float_log", 1e-3, 0.3),
        }
    elif "CatBoost" in name:
        return {
            "iterations": ("int", 50, 300),
            "depth": ("int", 4, 10),
            "learning_rate": ("float_log", 1e-3, 0.3),
        }
    return {}


class TrainingAgent(BaseAgent):
    name = "training_agent"
    description = "Trains ML models with Optuna HPO and cross-validation"
    stage = PipelineStage.TRAINING
    max_retries = 1

    def execute(self, state: AgentState) -> AgentState:
        processed_path = state.get("processed_data_path")
        if not processed_path or not Path(processed_path).exists():
            raise AgentError("No feature-engineered data found.")

        df = pd.read_parquet(processed_path)
        target_col = state.get("target_column", "")
        task_type = state.get("task_type", "regression")
        model_names = state.get("selected_model_types", [])
        settings = get_settings()
        device = settings.effective_training_device

        if not model_names:
            raise AgentError("No models selected. Run model selection first.")
        if not target_col or target_col not in df.columns:
            raise AgentError(f"Target column '{target_col}' not in data.")

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        # CV splitter
        is_classification = "classification" in task_type
        cv_folds = settings.cross_val_folds
        if is_classification:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = "roc_auc" if "binary" in task_type else "accuracy"
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = "r2"

        trained_models: list[dict[str, Any]] = []
        best_score = -np.inf
        best_model_name = ""
        best_model_path = ""
        total_start = time.perf_counter()

        # MLflow tracking
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
        except Exception:
            pass

        models_dir = Path(settings.reports_dir) / state.get("experiment_id", "default") / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        for model_name in model_names:
            self.log_action("training_model", model=model_name)
            start = time.perf_counter()

            try:
                # Optuna HPO
                best_params = self._run_optuna(
                    model_name, task_type, X, y, cv, scoring, device,
                    n_trials=min(settings.optuna_n_trials, 15),
                )

                # Train final model with best params
                model = _build_model(model_name, best_params, task_type, device)
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                model.fit(X, y)

                # Save model
                model_path = str(models_dir / f"{model_name}.joblib")
                joblib.dump(model, model_path)

                elapsed = time.perf_counter() - start
                mean_score = float(cv_scores.mean())

                result = {
                    "model_name": model_name,
                    "model_type": model_name,
                    "params": best_params,
                    "metrics": {scoring: round(mean_score, 4)},
                    "cv_scores": cv_scores.round(4).tolist(),
                    "cv_mean": round(mean_score, 4),
                    "cv_std": round(float(cv_scores.std()), 4),
                    "training_time_seconds": round(elapsed, 2),
                    "model_path": model_path,
                }
                trained_models.append(result)

                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = model_name
                    best_model_path = model_path

                # MLflow log
                try:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        mlflow.log_params(best_params)
                        mlflow.log_metric(scoring, mean_score)
                        mlflow.log_metric("cv_std", float(cv_scores.std()))
                        mlflow.log_metric("training_time_s", elapsed)
                except Exception:
                    pass

                self.logger.info(
                    "model_trained",
                    model=model_name,
                    score=mean_score,
                    time_s=round(elapsed, 2),
                )

            except Exception as e:
                self.logger.warning("model_training_failed", model=model_name, error=str(e))
                trained_models.append({"model_name": model_name, "error": str(e)})

        total_elapsed = time.perf_counter() - total_start

        state = dict(state)  # type: ignore[assignment]
        state["trained_models"] = trained_models
        state["best_model_name"] = best_model_name
        state["best_model_path"] = best_model_path
        state["training_duration_seconds"] = round(total_elapsed, 2)

        self.logger.info(
            "training_complete",
            n_models=len(trained_models),
            best=best_model_name,
            best_score=best_score,
        )
        return state  # type: ignore[return-value]

    def _run_optuna(
        self, model_name: str, task_type: str, X: np.ndarray,
        y: np.ndarray, cv: Any, scoring: str, device: str, n_trials: int
    ) -> dict[str, Any]:
        """Run Optuna HPO and return best params."""
        param_space = _get_param_space(model_name, task_type)
        if not param_space:
            return {}

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {}
            for pname, spec in param_space.items():
                if spec[0] == "int":
                    params[pname] = trial.suggest_int(pname, spec[1], spec[2])
                elif spec[0] == "float":
                    params[pname] = trial.suggest_float(pname, spec[1], spec[2])
                elif spec[0] == "float_log":
                    params[pname] = trial.suggest_float(pname, spec[1], spec[2], log=True)
                elif spec[0] == "categorical":
                    params[pname] = trial.suggest_categorical(pname, spec[1])
            try:
                model = _build_model(model_name, params, task_type, device)
                scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=-1)
                return float(scores.mean())
            except Exception:
                return -999.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=60)
        return study.best_params if study.best_params else {}

    def compute_confidence(self, state: AgentState) -> float:
        models = state.get("trained_models", [])
        if not models:
            return 0.0
        successful = [m for m in models if "error" not in m]
        return len(successful) / len(models)

    def _success_message(self, state: AgentState) -> str:
        models = state.get("trained_models", [])
        best = state.get("best_model_name", "None")
        return (
            f"✅ Training complete: {len(models)} models trained. "
            f"Best: **{best}**. Duration: {state.get('training_duration_seconds', 0):.1f}s."
        )
