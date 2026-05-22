"""
Memory Agent
=============
Manages long-term experiment memory using ChromaDB:
- Store experiment results as vector embeddings
- Semantic similarity search over past experiments
- RAG-style retrieval for experiment comparison
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.helpers import now_iso, safe_json_serialize


class MemoryAgent(BaseAgent):
    name = "memory_agent"
    description = "Stores and retrieves experiment memory using ChromaDB"
    stage = PipelineStage.MEMORY
    max_retries = 1

    def execute(self, state: AgentState) -> AgentState:
        try:
            from app.memory.experiment_memory import ExperimentMemory
            memory = ExperimentMemory()
        except Exception as e:
            self.logger.warning("memory_init_failed", error=str(e))
            state = dict(state)  # type: ignore[assignment]
            state["memory_stored"] = False
            state["similar_experiments"] = []
            return state  # type: ignore[return-value]

        exp_id = state.get("experiment_id", "")
        dataset_info = state.get("dataset_info", {})
        leaderboard = state.get("leaderboard", [])

        # Build experiment document for storage
        best_metrics = leaderboard[0]["metrics"] if leaderboard else {}
        document = (
            f"Dataset: {dataset_info.get('filename', 'unknown')} "
            f"({dataset_info.get('n_rows', 0)} rows, {dataset_info.get('n_cols', 0)} cols). "
            f"Task: {state.get('task_type', 'unknown')}. "
            f"Target: {state.get('target_column', 'unknown')}. "
            f"Best model: {state.get('best_model_name', 'none')}. "
            f"Metrics: {best_metrics}. "
            f"Features: {state.get('n_features_selected', 0)} selected. "
            f"Insights: {' '.join(state.get('eda_insights', [])[:3])}"
        )

        metadata = safe_json_serialize({
            "experiment_id": exp_id,
            "filename": dataset_info.get("filename"),
            "task_type": state.get("task_type"),
            "target_column": state.get("target_column"),
            "n_rows": dataset_info.get("n_rows"),
            "best_model": state.get("best_model_name"),
            "best_metrics": json.dumps(best_metrics),
            "timestamp": now_iso(),
        })

        # Store in ChromaDB
        stored = False
        try:
            memory.store_experiment(exp_id, document, metadata)
            stored = True
            self.log_action("experiment_stored", exp_id=exp_id)
        except Exception as e:
            self.logger.warning("memory_store_failed", error=str(e))

        # Retrieve similar experiments
        similar: list[dict[str, Any]] = []
        try:
            results = memory.find_similar(document, n_results=3)
            similar = [
                r for r in results
                if r.get("metadata", {}).get("experiment_id") != exp_id
            ]
            self.log_action("similar_experiments_found", count=len(similar))
        except Exception as e:
            self.logger.warning("memory_retrieve_failed", error=str(e))

        state = dict(state)  # type: ignore[assignment]
        state["memory_stored"] = stored
        state["similar_experiments"] = similar
        state["pipeline_complete"] = True

        self.logger.info("memory_complete", stored=stored, similar=len(similar))
        return state  # type: ignore[return-value]

    def _success_message(self, state: AgentState) -> str:
        similar = len(state.get("similar_experiments", []))
        return f"✅ Memory: experiment stored. Found {similar} similar past experiments."
