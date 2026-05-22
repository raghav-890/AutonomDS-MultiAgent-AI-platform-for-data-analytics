"""
AutonomDS Base Agent
=====================
Abstract base class for all 11 agents in the pipeline.
Provides LLM access, reflection capability, confidence scoring,
structured logging, execution tracking, and retry logic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import timezone, datetime
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.orchestration.state import AgentState, AgentExecutionRecord, AgentStatus, PipelineStage
from app.utils.config import get_settings, LLMProvider
from app.utils.helpers import now_iso
from app.utils.logger import get_logger


class AgentError(Exception):
    """Base exception for agent failures."""
    pass


class LLMUnavailableError(AgentError):
    """Raised when no LLM backend is reachable."""
    pass


def _build_llm(model_override: Optional[str] = None) -> BaseChatModel:
    """
    Build a LangChain LLM based on configured provider.
    Auto-detects Ollama availability, falls back to HuggingFace.
    """
    settings = get_settings()
    provider = settings.effective_llm_provider
    model = model_override or settings.ollama_model

    if provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=settings.ollama_temperature,
            timeout=settings.ollama_timeout,
        )
    else:
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.huggingface_model,
            huggingfacehub_api_token=settings.huggingface_api_token,
            task="text-generation",
            max_new_tokens=2048,
            temperature=0.1,
        )
        return ChatHuggingFace(llm=endpoint)


class BaseAgent(ABC):
    """
    Abstract base for all AutonomDS agents.

    Each concrete agent implements `execute()` which contains the
    core business logic. The `run()` method wraps `execute()` with:
    - Execution tracking
    - Retry logic (via tenacity)
    - Reflection capability
    - Confidence scoring
    - Structured logging
    """

    # Subclasses must define these
    name: str = "base_agent"
    description: str = "Base agent"
    stage: PipelineStage = PipelineStage.INGESTION
    max_retries: int = 2

    def __init__(self, model_override: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.logger = get_logger(self.name)
        self._llm: Optional[BaseChatModel] = None
        self._model_override = model_override

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize LLM on first access."""
        if self._llm is None:
            self._llm = _build_llm(self._model_override)
        return self._llm

    # ── Core Interface ────────────────────────────────────────────────────────

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """
        Agent's core logic. Reads from state, returns updated state.
        Must be implemented by every concrete agent.
        """
        ...

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the agent with full lifecycle management:
        tracking, retry, reflection, confidence scoring.
        """
        record = AgentExecutionRecord(
            agent_name=self.name,
            stage=self.stage,
            status=AgentStatus.RUNNING,
            started_at=now_iso(),
        )
        self.logger.info("agent_started", stage=self.stage.value)

        # Update state tracking
        state = dict(state)  # type: ignore[assignment]
        state.setdefault("agent_executions", [])
        state.setdefault("errors", [])
        state.setdefault("messages", [])
        state["current_agent"] = self.name
        state["current_stage"] = self.stage.value

        start_time = time.perf_counter()

        try:
            state = self._run_with_retry(state)
            record.status = AgentStatus.COMPLETED
            record.confidence = self.compute_confidence(state)

            # Append NL message to conversation history
            state["messages"].append({
                "role": "agent",
                "agent": self.name,
                "content": self._success_message(state),
                "timestamp": now_iso(),
            })

            # Trigger reflection if confidence is low
            if record.confidence < 0.6:
                self.logger.warning(
                    "low_confidence_triggering_reflection",
                    confidence=record.confidence,
                )
                state["should_reflect"] = True
                state.setdefault("reflection_notes", []).append(
                    f"{self.name}: low confidence ({record.confidence:.2f})"
                )

        except Exception as e:
            record.status = AgentStatus.FAILED
            record.error = str(e)
            state["errors"].append(f"{self.name}: {e}")
            state["messages"].append({
                "role": "agent",
                "agent": self.name,
                "content": f"❌ {self.name} failed: {e}",
                "timestamp": now_iso(),
            })
            self.logger.error("agent_failed", error=str(e), exc_info=True)
            raise AgentError(f"{self.name} failed: {e}") from e

        finally:
            elapsed = time.perf_counter() - start_time
            record.completed_at = now_iso()
            record.duration_seconds = round(elapsed, 3)
            state["agent_executions"].append(record.to_dict())
            self.logger.info(
                "agent_finished",
                status=record.status.value,
                duration_s=record.duration_seconds,
                confidence=record.confidence,
            )

        return state  # type: ignore[return-value]

    def _run_with_retry(self, state: AgentState) -> AgentState:
        """Execute with tenacity retry on transient failures."""
        attempts = 0
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return self.execute(state)
            except AgentError:
                raise  # Don't retry logic errors
            except Exception as e:
                attempts = attempt + 1
                last_error = e
                self.logger.warning(
                    "agent_retry",
                    attempt=attempts,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

        raise last_error or Exception("Unknown error after retries")

    # ── LLM Utilities ─────────────────────────────────────────────────────────

    def ask_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response as a string.
        Handles both ChatModel and LLM interfaces.
        """
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            self.logger.warning("llm_call_failed", error=str(e))
            return f"[LLM unavailable: {e}]"

    def reflect(self, state: AgentState, critique_context: str) -> str:
        """
        Self-reflection: ask LLM to critique this agent's output
        and suggest improvements.
        """
        system = (
            f"You are an expert AI system reviewing the work of the {self.name} agent. "
            "Provide a concise, actionable critique. Be specific. "
            "If the work is good, say so briefly."
        )
        user = (
            f"Review the following output from {self.name}:\n\n"
            f"{critique_context}\n\n"
            "Provide: 1) What was done well, 2) What could be improved, "
            "3) A confidence score 0.0-1.0 for this work."
        )
        return self.ask_llm(system, user)

    # ── Confidence Scoring ────────────────────────────────────────────────────

    def compute_confidence(self, state: AgentState) -> float:
        """
        Compute a confidence score (0.0–1.0) for this agent's output.
        Default: 1.0 if no errors, 0.3 if errors exist.
        Subclasses should override with domain-specific logic.
        """
        errors = state.get("errors", [])
        agent_errors = [e for e in errors if self.name in e]
        if agent_errors:
            return 0.3
        return 1.0

    def _success_message(self, state: AgentState) -> str:
        """Default success message. Subclasses can override."""
        return f"✅ {self.name} completed successfully."

    # ── Convenience Methods ───────────────────────────────────────────────────

    def log_action(self, action: str, **kwargs: Any) -> None:
        """Log a specific action taken by this agent."""
        self.logger.info("agent_action", action=action, **kwargs)

    def emit_insight(self, state: AgentState, insight: str) -> None:
        """Append an insight to the EDA insights list."""
        state.setdefault("eda_insights", [])
        state["eda_insights"].append(f"[{self.name}] {insight}")

    def update_stage(self, state: AgentState, stage: PipelineStage) -> None:
        """Update the current pipeline stage in state."""
        state["current_stage"] = stage.value
