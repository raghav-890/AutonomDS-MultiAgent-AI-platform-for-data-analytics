# AutonomDS — Complete Implementation Plan

## Current State Assessment

The project has a **strong foundation** from the previous session with significant portions already built. The goal now is to **complete all missing pieces, fix integration issues, and produce a fully runnable, end-to-end system**.

### ✅ Already Built (Previous Session)
| Component | File | Status |
|---|---|---|
| Base Agent | `app/agents/base_agent.py` | ✅ Complete |
| Ingestion Agent | `app/agents/ingestion_agent.py` | ✅ Complete |
| Cleaning Agent | `app/agents/cleaning_agent.py` | ✅ Complete |
| Feature Engineering Agent | `app/agents/feature_engineering_agent.py` | ✅ Complete |
| Model Selection Agent | `app/agents/model_selection_agent.py` | ✅ Complete |
| Training Agent | `app/agents/training_agent.py` | ✅ Complete |
| Evaluation Agent | `app/agents/evaluation_agent.py` | ✅ Complete |
| Explainability Agent | `app/agents/explainability_agent.py` | ✅ Complete |
| Report Agent | `app/agents/report_agent.py` | ✅ Complete |
| Memory Agent | `app/agents/memory_agent.py` | ✅ Complete |
| LangGraph Orchestration | `app/orchestration/graph.py` | ✅ Complete |
| Agent State TypedDict | `app/orchestration/state.py` | ✅ Complete |
| ChromaDB Store | `app/memory/chroma_store.py` | ✅ Complete |
| Embeddings | `app/memory/embeddings.py` | ✅ Complete |
| Experiment Memory | `app/memory/experiment_memory.py` | ✅ Complete |
| FastAPI App | `app/api/main.py` | ✅ Complete |
| Upload Route | `app/api/routes/upload.py` | ✅ Complete |
| Pipeline Route | `app/api/routes/pipeline.py` | ✅ Complete |
| Experiments Route | `app/api/routes/experiments.py` | ✅ Complete |
| Reports Route | `app/api/routes/reports.py` | ✅ Complete |
| Streamlit Main App | `app/frontend/streamlit_app.py` | ✅ Complete |
| Home Page | `app/frontend/pages/home.py` | ✅ Complete |
| Upload Page | `app/frontend/pages/upload_page.py` | ✅ Complete |
| Pipeline Page | `app/frontend/pages/pipeline_page.py` | ✅ Complete |
| Models Page | `app/frontend/pages/models_page.py` | ✅ Complete |
| Experiments Page | `app/frontend/pages/experiments_page.py` | ✅ Complete |
| Reports Page | `app/frontend/pages/reports_page.py` | ✅ Complete |
| Dark Mode CSS Theme | `app/frontend/styles/theme.py` | ✅ Complete |
| Config / Settings | `app/utils/config.py` | ✅ Complete |
| Helpers | `app/utils/helpers.py` | ✅ Complete |
| Logger | `app/utils/logger.py` | ✅ Complete |
| Validators | `app/utils/validators.py` | ✅ Complete |
| API Schemas | `app/api/schemas.py` | ✅ Complete |
| Docker Compose | `docker-compose.yml` | ✅ Complete |
| Dockerfiles (3) | `docker/` | ✅ Complete |
| Render Config | `deployment/render.yaml` | ✅ Complete |
| CI/CD | `deployment/.github/workflows/ci.yml` | ✅ Complete |
| Requirements | `requirements.txt` | ✅ Complete |
| README | `README.md` | ✅ Partial |
| Basic Tests | `tests/unit/` (3 files) | ✅ Partial |

---

### ❌ Missing / Incomplete Components

#### Critical Missing Agents
1. **EDA Agent** — `app/agents/eda_agent.py` — **NOT CREATED** (imported in `graph.py` but missing!)
2. **EDA Page** — `app/frontend/pages/eda_page.py` — Stub only (1.7KB, needs full implementation)

#### Missing Infrastructure Files
3. **`app/utils/__init__.py`** — Package inits exist but need checking
4. **`app/rag/`** directory — Not created (RAG system referenced in spec but not wired)
5. **`app/monitoring/`** directory — Not created (MLflow tracking not wired)
6. **`app/tools/`** directory — Not created 
7. **`app/models/`** directory — Not created
8. **`notebooks/`** directory — Not created
9. **`docs/`** directory with architecture docs — Not created
10. **`datasets/`** directory — Exists but empty

#### Missing Tests
11. `tests/unit/test_cleaning_agent.py`
12. `tests/unit/test_training_agent.py`
13. `tests/unit/test_memory.py`
14. `tests/integration/test_api.py`
15. `tests/integration/test_pipeline.py`

#### Missing Deployment Files
16. **`deployment/.github/workflows/ci.yml`** — Needs to be moved to correct `.github/` location
17. **`railway.json`** — Railway deployment config
18. **`Makefile`** — Developer convenience commands

#### Missing Documentation
19. `docs/architecture.md`
20. `docs/agent_design.md`
21. `docs/rag_system.md`
22. `docs/deployment.md`

#### Quality Issues to Fix
23. `app/frontend/pages/eda_page.py` — Stub needs full rich implementation
24. `docker/Dockerfile.api` — Check CMD and paths
25. `README.md` — Missing badges, complete setup guide
26. `.env.example` — Verify all vars documented

---

## Proposed Changes

### Phase 1: Critical — EDA Agent (BLOCKER)

#### [NEW] `app/agents/eda_agent.py`
The most critical missing file. The entire graph will fail to import without this.
- Missing values analysis with heatmaps
- Correlation matrix (Plotly)
- Outlier detection (IQR-based)
- Class balance analysis
- Skewness analysis
- LLM-generated natural language insights
- Chart persistence to disk

---

### Phase 2: Complete EDA Frontend Page

#### [MODIFY] `app/frontend/pages/eda_page.py`
Currently a 1.7KB stub. Needs full implementation:
- Dataset overview metrics
- Interactive correlation heatmap display
- Missing values chart
- Distribution plots
- Class balance visualization
- LLM insights display

---

### Phase 3: RAG Directory + Monitoring

#### [NEW] `app/rag/__init__.py`
#### [NEW] `app/rag/retrieval.py`
Advanced RAG pipeline that wraps ExperimentMemory with:
- Query classification
- Context-aware retrieval
- Augmented prompt construction
- Support for KnowledgeRetrievalAgent

#### [NEW] `app/monitoring/__init__.py`
#### [NEW] `app/monitoring/mlflow_tracker.py`
MLflow integration:
- Experiment tracking
- Metric logging
- Artifact logging
- Run management

#### [NEW] `app/tools/__init__.py`
#### [NEW] `app/tools/data_tools.py`
Reusable data utilities for agents.

---

### Phase 4: Missing Directory Scaffolding

#### [NEW] `app/models/__init__.py`
#### [NEW] `notebooks/exploratory.ipynb` placeholder
#### [NEW] `datasets/.gitkeep`

---

### Phase 5: Complete Test Suite

#### [NEW] `tests/unit/test_cleaning_agent.py`
#### [NEW] `tests/unit/test_training_agent.py`
#### [NEW] `tests/unit/test_memory.py`
#### [NEW] `tests/integration/test_api.py`
#### [NEW] `tests/integration/test_pipeline.py`
#### [MODIFY] `tests/unit/test_eda_agent.py` — Expand from stub

---

### Phase 6: Developer Infrastructure

#### [NEW] `Makefile`
Convenient targets: `make dev`, `make test`, `make docker-up`, `make lint`

#### [NEW] `railway.json`
Railway deployment configuration

#### [MODIFY] `.github/workflows/ci.yml`
Ensure CI pipeline is in the correct location

#### [MODIFY] `README.md`
Complete with badges, architecture diagram in ASCII, and full setup guide

---

### Phase 7: Documentation

#### [NEW] `docs/architecture.md`
System design, layer diagram, component overview

#### [NEW] `docs/agent_design.md`
Each agent's responsibilities, state mutations, confidence scoring

#### [NEW] `docs/rag_system.md`
RAG pipeline, ChromaDB integration, memory retrieval

#### [NEW] `docs/deployment.md`
Docker, Render, Railway, HuggingFace Spaces deployment guides

---

## Open Questions

> [!IMPORTANT]
> **EDA Agent Dependency**: The `graph.py` currently imports `from app.agents.eda_agent import EDAAgent` — this file is completely missing. This is a **hard blocker** that prevents the entire system from running. Creating the EDA agent is the highest priority.

> [!NOTE]
> **GitHub Actions Location**: The CI file is at `deployment/.github/workflows/ci.yml` but GitHub requires it at `.github/workflows/ci.yml` at the repo root. I will create the properly-located file.

> [!NOTE]
> **Celery/Redis**: The system falls back to sync mode when Redis isn't available, so local development works without Docker. The async Celery path is wired but optional.

---

## Verification Plan

### Automated Tests
```bash
# After implementation
cd autonomous-ds-agent
python -m pytest tests/ -v --tb=short
```

### Import Validation
```bash
python -c "from app.agents.eda_agent import EDAAgent; print('EDA Agent OK')"
python -c "from app.orchestration.graph import pipeline_graph; print('Graph compiled OK')"
python -c "from app.api.main import app; print('FastAPI app OK')"
```

### Manual Verification
- Run `uvicorn app.api.main:app --reload --port 8000` → check `/health` and `/docs`
- Run `streamlit run app/frontend/streamlit_app.py` → check all 7 pages load
- Upload a CSV → trigger pipeline → confirm graph compiles and runs

---

## Priority Execution Order

| Priority | Task | Effort |
|---|---|---|
| P0 🔴 | Create `eda_agent.py` (BLOCKER) | Large |
| P0 🔴 | Expand `eda_page.py` from stub | Medium |
| P1 🟡 | RAG module (`app/rag/`) | Medium |
| P1 🟡 | MLflow monitoring module | Small |
| P1 🟡 | Tools + Models dirs | Small |
| P2 🟢 | Complete test suite | Large |
| P2 🟢 | Makefile + Railway config | Small |
| P2 🟢 | Fix GitHub Actions location | Small |
| P3 🔵 | Documentation (4 docs files) | Medium |
| P3 🔵 | README completion | Medium |
