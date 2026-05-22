"""
LangGraph Orchestration Graph
===============================
Defines the full multi-agent pipeline as a LangGraph StateGraph.
Includes conditional routing, reflection loops, retry logic,
and SQLite-backed state persistence.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.ingestion_agent import DataIngestionAgent
from app.agents.eda_agent import EDAAgent
from app.agents.cleaning_agent import DataCleaningAgent
from app.agents.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.model_selection_agent import ModelSelectionAgent
from app.agents.training_agent import TrainingAgent
from app.agents.evaluation_agent import EvaluationAgent
from app.agents.explainability_agent import ExplainabilityAgent
from app.agents.report_agent import ReportAgent
from app.agents.memory_agent import MemoryAgent
from app.orchestration.state import AgentState, PipelineStage
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("orchestrator")
settings = get_settings()

# ── Agent Singletons ──────────────────────────────────────────────────────────
_ingestion = DataIngestionAgent()
_eda = EDAAgent()
_cleaning = DataCleaningAgent()
_feature_eng = FeatureEngineeringAgent()
_model_sel = ModelSelectionAgent()
_training = TrainingAgent()
_evaluation = EvaluationAgent()
_explainability = ExplainabilityAgent()
_report = ReportAgent()
_memory = MemoryAgent()


# ── Node Functions ─────────────────────────────────────────────────────────────

def node_ingest(state: AgentState) -> AgentState:
    return _ingestion.run(state)

def node_eda(state: AgentState) -> AgentState:
    return _eda.run(state)

def node_clean(state: AgentState) -> AgentState:
    return _cleaning.run(state)

def node_feature_eng(state: AgentState) -> AgentState:
    return _feature_eng.run(state)

def node_model_select(state: AgentState) -> AgentState:
    return _model_sel.run(state)

def node_train(state: AgentState) -> AgentState:
    return _training.run(state)

def node_evaluate(state: AgentState) -> AgentState:
    return _evaluation.run(state)

def node_explain(state: AgentState) -> AgentState:
    return _explainability.run(state)

def node_report(state: AgentState) -> AgentState:
    return _report.run(state)

def node_memory(state: AgentState) -> AgentState:
    return _memory.run(state)

def node_reflect(state: AgentState) -> AgentState:
    """Reflection node: critique pipeline progress and suggest retry."""
    logger.info("reflection_triggered")
    notes = state.get("reflection_notes", [])
    errors = state.get("errors", [])

    # Simple reflection: LLM reviews issues
    from app.agents.base_agent import _build_llm
    try:
        llm = _build_llm()
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=(
                "You are an AI pipeline reviewer. Analyze these issues and decide "
                "if the pipeline should retry the problematic stage or continue. "
                "Reply with JSON: {\"action\": \"retry\" | \"continue\", \"stage\": \"<stage_name>\", \"reason\": \"...\"}"
            )),
            HumanMessage(content=f"Notes: {notes}\nErrors: {errors}"),
        ]
        response = llm.invoke(messages).content
        import re, json
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        if match:
            decision = json.loads(match.group())
            logger.info("reflection_decision", decision=decision)
    except Exception as e:
        logger.warning("reflection_llm_failed", error=str(e))

    # Reset reflection flag and continue
    state = dict(state)  # type: ignore[assignment]
    state["should_reflect"] = False
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state  # type: ignore[return-value]


# ── Routing Functions ──────────────────────────────────────────────────────────

def route_after_ingest(state: AgentState) -> Literal["eda", "end"]:
    if state.get("errors") and state.get("dataset_info") is None:
        logger.warning("routing_to_end_ingestion_failed")
        return "end"
    return "eda"

def route_after_evaluate(state: AgentState) -> Literal["explain", "reflect", "end"]:
    if state.get("should_reflect") and state.get("retry_count", 0) < 2:
        return "reflect"
    if not state.get("leaderboard"):
        return "end"
    return "explain"

def route_after_reflect(state: AgentState) -> Literal["train", "explain", "end"]:
    """After reflection, retry training or proceed."""
    retry_count = state.get("retry_count", 0)
    if retry_count >= 3:
        return "end"
    return "train"  # Default: retry training with different settings


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_pipeline_graph(use_checkpointing: bool = True) -> Any:
    """
    Build and compile the full LangGraph pipeline.

    Args:
        use_checkpointing: If True, use SQLite checkpointing for state persistence.

    Returns:
        Compiled LangGraph graph.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ────────────────────────────────────────────────────────
    graph.add_node("ingest", node_ingest)
    graph.add_node("eda", node_eda)
    graph.add_node("clean", node_clean)
    graph.add_node("feature_eng", node_feature_eng)
    graph.add_node("model_select", node_model_select)
    graph.add_node("train", node_train)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("reflect", node_reflect)
    graph.add_node("explain", node_explain)
    graph.add_node("report", node_report)
    graph.add_node("memory", node_memory)

    # ── Add edges ─────────────────────────────────────────────────────────
    graph.add_edge(START, "ingest")
    graph.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {"eda": "eda", "end": END},
    )
    graph.add_edge("eda", "clean")
    graph.add_edge("clean", "feature_eng")
    graph.add_edge("feature_eng", "model_select")
    graph.add_edge("model_select", "train")
    graph.add_edge("train", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"explain": "explain", "reflect": "reflect", "end": END},
    )
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"train": "train", "explain": "explain", "end": END},
    )
    graph.add_edge("explain", "report")
    graph.add_edge("report", "memory")
    graph.add_edge("memory", END)

    # ── Compile ──────────────────────────────────────────────────────────
    if use_checkpointing:
        db_path = settings.upload_dir.parent / "experiments" / "checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string(str(db_path))
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# ── Module-level compiled graph ────────────────────────────────────────────────
try:
    pipeline_graph = build_pipeline_graph(use_checkpointing=True)
    logger.info("pipeline_graph_compiled")
except Exception as e:
    logger.warning("graph_compile_failed_no_checkpointing", error=str(e))
    pipeline_graph = build_pipeline_graph(use_checkpointing=False)
