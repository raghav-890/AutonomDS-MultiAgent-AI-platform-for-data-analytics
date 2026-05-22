# Autonomous Data Science Research Agent — Implementation Plan

> **Platform codename:** `AutonomDS` — A startup-grade, fully free, multi-agent autonomous data science platform.

---

## Overview

We are building a **multi-agent AI system** that acts as an autonomous junior data scientist. A user uploads a dataset; the platform autonomously performs EDA, cleaning, feature engineering, model selection, training, evaluation, explanation, and report generation — all orchestrated by a LangGraph workflow, powered by local LLMs via Ollama.

The final deliverable is a **monorepo** that is Docker-composable, deployable on free tiers (Render, Railway, HuggingFace Spaces), and has a beautiful Streamlit frontend.

---

## Architecture Decision Record (ADR)

### Why Streamlit + FastAPI (not pure FastAPI + React)?
- Streamlit gives us a **data-native UI** with zero frontend build tooling
- FastAPI sits behind as the **agent orchestration API** layer, handling long-running Celery tasks
- This separation lets the Streamlit app remain stateless while heavy computation runs async

### Why LangGraph over pure LangChain?
- LangGraph gives us **graph-based state machines** — critical for retry loops, conditional routing, and reflection
- Each agent is a LangGraph node; the orchestrator is the graph controller
- State is persisted via SQLite checkpointer (built into LangGraph)

### Why Ollama for LLMs?
- 100% free, runs locally, no rate limits
- Supports Mistral 7B, Llama 3 8B, DeepSeek Coder, Phi-3 Mini
- HuggingFace transformers as fallback for embeddings + specialized models

### Why ChromaDB?
- Embedded, no server needed, persistent on disk
- Perfect for semantic memory over experiment history
- sentence-transformers for embeddings (free, local)

### Why Celery + Redis?
- LLM inference + ML training can take minutes
- Celery decouples frontend from long-running tasks
- Redis (free, in Docker) acts as broker + result backend

### Why MLflow?
- Best free experiment tracking
- Artifacts, metrics, and model registry all in one
- SQLite backend = zero infra cost

---

## Phased Build Plan

### Phase 1 — MVP (Core Pipeline)
- Project skeleton, Docker setup, config system
- Data Ingestion Agent
- EDA Agent  
- Basic Streamlit UI (upload + EDA display)
- FastAPI health endpoint

### Phase 2 — Multi-Agent Pipeline
- Data Cleaning Agent
- Feature Engineering Agent
- Model Selection Agent
- Training Agent
- Evaluation Agent
- LangGraph orchestration graph connecting all agents
- Full Streamlit pipeline dashboard

### Phase 3 — Memory + Explainability
- Explainability Agent (SHAP + LIME)
- Report Generation Agent (PDF + Markdown)
- Memory Agent (ChromaDB + semantic search)
- Experiment history UI

### Phase 4 — Advanced Orchestration
- Reflection loops (agent critiques its own outputs)
- Confidence scoring
- Failure recovery + retry logic
- Celery async task queue integration
- MLflow experiment tracking

### Phase 5 — Deployment
- Docker Compose (all services)
- GitHub Actions CI/CD
- Render / Railway deployment configs
- HuggingFace Spaces deployment
- Environment variable management

### Phase 6 — Production Polish
- Comprehensive unit tests
- Logging + monitoring
- Security hardening (file validation, sandboxed execution)
- README with badges + architecture diagram
- Example datasets

---

## Proposed Project Structure

```
autonomous-ds-agent/
│
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Abstract base for all agents
│   │   ├── ingestion_agent.py
│   │   ├── eda_agent.py
│   │   ├── cleaning_agent.py
│   │   ├── feature_engineering_agent.py
│   │   ├── model_selection_agent.py
│   │   ├── training_agent.py
│   │   ├── evaluation_agent.py
│   │   ├── explainability_agent.py
│   │   ├── report_agent.py
│   │   └── memory_agent.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app entry
│   │   ├── routes/
│   │   │   ├── upload.py
│   │   │   ├── pipeline.py
│   │   │   ├── experiments.py
│   │   │   └── reports.py
│   │   └── schemas.py              # Pydantic models
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── graph.py                # LangGraph workflow definition
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── router.py               # Conditional routing logic
│   │   └── checkpointer.py        # SQLite state persistence
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── chroma_store.py         # ChromaDB vector store
│   │   ├── experiment_memory.py    # Experiment retrieval + storage
│   │   └── embeddings.py           # sentence-transformers
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py             # Model catalog
│   │   ├── trainer.py              # Training loop abstraction
│   │   └── evaluator.py            # Metrics computation
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── python_repl.py          # Sandboxed Python execution
│   │   ├── sql_tool.py             # SQLite query tool
│   │   ├── viz_tool.py             # Plotly chart generation
│   │   └── file_tool.py            # File I/O utilities
│   │
│   ├── frontend/
│   │   ├── streamlit_app.py        # Main Streamlit entry
│   │   ├── pages/
│   │   │   ├── 01_Upload.py
│   │   │   ├── 02_EDA.py
│   │   │   ├── 03_Pipeline.py
│   │   │   ├── 04_Models.py
│   │   │   ├── 05_Experiments.py
│   │   │   └── 06_Reports.py
│   │   ├── components/
│   │   │   ├── agent_monitor.py
│   │   │   ├── metrics_cards.py
│   │   │   ├── workflow_viz.py
│   │   │   └── chat_interface.py
│   │   └── styles/
│   │       └── theme.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── mlflow_tracker.py
│   │   └── logger.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Pydantic Settings
│       ├── validators.py           # File validation
│       └── helpers.py
│
├── datasets/                        # Sample datasets
│   ├── titanic.csv
│   └── boston_housing.csv
│
├── experiments/                     # MLflow + ChromaDB storage
├── notebooks/                       # Jupyter exploration notebooks
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── Dockerfile.worker
│
├── docs/
│   └── architecture.md
│
├── deployment/
│   ├── render.yaml
│   ├── railway.toml
│   └── .github/workflows/ci.yml
│
├── .env.example
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Proposed Changes (Files to Create)

### Foundation Layer

#### [NEW] `pyproject.toml`
Python project metadata, tool configs (ruff, pytest, mypy).

#### [NEW] `requirements.txt`
All dependencies pinned. Key packages:
- `langgraph`, `langchain`, `langchain-community`, `langchain-ollama`
- `fastapi`, `uvicorn`, `celery`, `redis`
- `streamlit`, `plotly`, `matplotlib`
- `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`
- `shap`, `lime`, `optuna`, `mlflow`
- `chromadb`, `sentence-transformers`
- `pandas`, `numpy`, `openpyxl`, `kaggle`
- `reportlab`, `weasyprint` (PDF generation)
- `pydantic-settings`

#### [NEW] `.env.example`
All environment variables documented.

#### [NEW] `docker-compose.yml`
Services: `api`, `frontend`, `worker`, `redis`, `mlflow`

---

### App Core

#### [NEW] `app/utils/config.py`
Pydantic BaseSettings — reads from `.env`, typed config.

#### [NEW] `app/utils/logger.py`
Structured logging with `structlog`.

#### [NEW] `app/orchestration/state.py`
`AgentState` TypedDict — the central state passed between all LangGraph nodes.

#### [NEW] `app/orchestration/graph.py`
The full LangGraph `StateGraph` — all 11 agents as nodes, conditional edges, retry logic.

---

### Agents (11 total)

#### [NEW] `app/agents/base_agent.py`
Abstract `BaseAgent` with: `name`, `description`, `run()`, `reflect()`, `confidence_score()`.

#### [NEW] `app/agents/ingestion_agent.py`
Handles CSV/Excel/SQLite/Kaggle. Uses `pandas` + schema validation.

#### [NEW] `app/agents/eda_agent.py`
Missing values, correlations, skewness, outliers, class imbalance. Generates Plotly charts + NL insights via LLM.

#### [NEW] `app/agents/cleaning_agent.py`
Imputation, encoding, scaling, duplicate removal, leakage detection.

#### [NEW] `app/agents/feature_engineering_agent.py`
Feature selection (RFECV), polynomial/interaction features, PCA, feature importance.

#### [NEW] `app/agents/model_selection_agent.py`
Task type inference (regression/classification/clustering), model candidate selection, baseline benchmarking.

#### [NEW] `app/agents/training_agent.py`
Cross-validation, Optuna hyperparameter tuning, early stopping, GPU support via PyTorch.

#### [NEW] `app/agents/evaluation_agent.py`
Full metrics suite, confusion matrix, ROC, residuals, model ranking leaderboard.

#### [NEW] `app/agents/explainability_agent.py`
SHAP TreeExplainer/KernelExplainer, LIME, feature importance plots, interpretability report.

#### [NEW] `app/agents/report_agent.py`
PDF report (ReportLab), Markdown report, downloadable experiment summary.

#### [NEW] `app/agents/memory_agent.py`
ChromaDB storage, semantic similarity search over past experiments, RAG retrieval.

---

### API Layer

#### [NEW] `app/api/main.py`
FastAPI app with CORS, lifespan, exception handlers.

#### [NEW] `app/api/routes/upload.py`
File upload with validation (type, size, schema check).

#### [NEW] `app/api/routes/pipeline.py`
Trigger full pipeline, poll status, get results.

#### [NEW] `app/api/routes/experiments.py`
List, compare, and retrieve past experiments.

#### [NEW] `app/api/routes/reports.py`
Download generated reports.

---

### Frontend

#### [NEW] `app/frontend/streamlit_app.py`
Main Streamlit multi-page app. Dark mode, custom CSS, sidebar nav.

#### [NEW] `app/frontend/pages/01_Upload.py` through `06_Reports.py`
Each page handles its own section of the workflow.

#### [NEW] `app/frontend/components/`
Reusable Streamlit components: agent activity monitor, metrics cards, workflow graph viz, chat interface.

---

### Memory

#### [NEW] `app/memory/chroma_store.py`
ChromaDB persistent client + collection management.

#### [NEW] `app/memory/experiment_memory.py`
Store/retrieve experiments with metadata and embeddings.

---

### Tools

#### [NEW] `app/tools/python_repl.py`
Sandboxed Python exec using `RestrictedPython`.

#### [NEW] `app/tools/viz_tool.py`
LangChain tool wrapper for Plotly chart generation.

---

### Docker + Deployment

#### [NEW] `docker/Dockerfile.api`
#### [NEW] `docker/Dockerfile.frontend`
#### [NEW] `docker/Dockerfile.worker`
#### [NEW] `docker-compose.yml`
#### [NEW] `deployment/render.yaml`
#### [NEW] `deployment/.github/workflows/ci.yml`

---

### Tests

#### [NEW] `tests/unit/test_ingestion_agent.py`
#### [NEW] `tests/unit/test_eda_agent.py`
#### [NEW] `tests/unit/test_orchestration.py`
#### [NEW] `tests/conftest.py`

---

### Documentation

#### [NEW] `README.md`
Professional README with badges, architecture diagram (Mermaid), setup instructions, demo guide, API docs, roadmap.

---

## Key Architecture Decisions

| Decision | Choice | Reason |
|---|---|---|
| LLM Backend | Ollama (Mistral 7B) | Free, local, no rate limits |
| Orchestration | LangGraph | Native state machines, retry, conditional routing |
| Vector DB | ChromaDB | Embedded, free, no server |
| Experiment Tracking | MLflow | Best free option, full artifacts support |
| Task Queue | Celery + Redis | Decouples UI from long ML jobs |
| PDF Reports | ReportLab | Pure Python, no browser dependency |
| Embeddings | sentence-transformers | Free, local, high quality |

---

## Verification Plan

### Automated Tests
```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Integration Test
- Upload `titanic.csv` → trigger full pipeline → verify all 11 agents complete → download PDF report

### Docker Smoke Test
```bash
docker compose up --build
curl http://localhost:8000/health
```

### UI Verification
- Browser subagent to screenshot Streamlit UI at each major page

---

## Open Questions

> [!IMPORTANT]
> **Ollama setup**: The platform assumes Ollama is running locally. On deployment targets (Render/Railway free tier), Ollama cannot run due to RAM constraints. I will use **HuggingFace Inference API (free tier)** as the cloud fallback, with `HuggingFaceEndpoint` in LangChain. The system will auto-detect the environment.

> [!NOTE]
> **GPU support**: Training will use `device="cuda"` if available, else `"cpu"`. No change to code needed — PyTorch handles this.

> [!NOTE]  
> **Kaggle integration**: Requires a `KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env`. The UI will prompt users to add these. This is optional and won't block other data sources.

> [!WARNING]
> **CatBoost on ARM Mac**: CatBoost sometimes has issues on Apple Silicon. I will pin a compatible version and add a fallback to skip CatBoost if unavailable.

---

*Implementation will proceed phase by phase. Each phase will be tracked in `task.md`.*
