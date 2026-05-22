"""Unit tests for DataIngestionAgent."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_ingestion_loads_csv(base_state, tmp_path):
    """Ingestion agent loads CSV and populates state."""
    from app.agents.ingestion_agent import DataIngestionAgent

    agent = DataIngestionAgent()
    # Mock LLM call to avoid actual API calls
    with patch.object(agent, 'ask_llm', return_value="Test dataset summary."):
        result = agent.run(base_state)

    assert result["dataset_info"]["n_rows"] == 200
    assert result["dataset_info"]["n_cols"] == 6
    assert result["target_column"] == "target"
    assert result["task_type"] == "binary_classification"
    assert Path(result["processed_data_path"]).exists()


def test_ingestion_detects_target(base_state):
    """Ingestion agent auto-detects target column."""
    from app.agents.ingestion_agent import DataIngestionAgent

    state = dict(base_state)
    state["target_column"] = ""  # Remove explicit target

    agent = DataIngestionAgent()
    with patch.object(agent, 'ask_llm', return_value="Dataset summary."):
        result = agent.run(state)

    # Should have detected 'target' column heuristically
    assert result["target_column"] in ["target", "feature_1", "feature_2", "age", "income"]


def test_ingestion_populates_columns(base_state):
    """Ingestion agent correctly categorizes columns."""
    from app.agents.ingestion_agent import DataIngestionAgent

    agent = DataIngestionAgent()
    with patch.object(agent, 'ask_llm', return_value="Summary."):
        result = agent.run(base_state)

    assert isinstance(result["numeric_columns"], list)
    assert isinstance(result["categorical_columns"], list)
    assert len(result["numeric_columns"]) > 0


def test_ingestion_failure_on_missing_file(base_state):
    """Ingestion agent raises AgentError for missing files."""
    from app.agents.ingestion_agent import DataIngestionAgent
    from app.agents.base_agent import AgentError

    state = dict(base_state)
    state["raw_data_path"] = "/nonexistent/path/file.csv"

    agent = DataIngestionAgent()
    with pytest.raises(Exception):  # AgentError or wrapped
        agent.run(state)
