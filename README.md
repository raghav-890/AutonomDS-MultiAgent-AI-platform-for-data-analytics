# AutonomDS — Autonomous Multi-Agent Data Science Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-FF6B6B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![CI](https://img.shields.io/github/actions/workflow/status/your-username/autonomds/ci.yml?style=for-the-badge&label=CI)

**An autonomous AI data science operating system composed of 11 specialised ML agents.**

*Upload any dataset → watch the agents collaborate → receive production-ready ML models, reports, and insights.*

[🚀 Quick Start](#quick-start) · [📖 Docs](docs/) · [🏗️ Architecture](#architecture) · [🤝 Contributing](#contributing)

</div>

---

## What Is AutonomDS?

AutonomDS is **not a chatbot**. It's a **multi-agent AI system** that behaves like a full data science team:

| Agent | Role |
|---|---|
| 🗂️ **DataIngestionAgent** | Loads CSV/Excel/Parquet/SQLite, infers schema + task type |
| 🔍 **EDAAgent** | Missing values, correlations, outliers, class balance, LLM insights |
| 🧹 **DataCleaningAgent** | Imputation, encoding, scaling, leakage detection |
| ⚙️ **FeatureEngineeringAgent** | RFECV selection, polynomial features, PCA |
| 🧠 **KnowledgeRetrievalAgent** | RAG over past experiments for better decisions |
| 🤔 **ModelSelectionAgent** | Recommends models based on task + past experiment memory |
| 🏋️ **TrainingAgent** | Optuna HPO, cross-validation, model persistence, MLflow tracking |
| 📊 **EvaluationAgent** | Full metrics leaderboard, confusion matrices, ROC curves |
| 🔬 **ExplainabilityAgent** | SHAP + LIME analysis, feature importance charts |
| 📄 **ReportAgent** | PDF + Markdown + JSON reports with LLM executive summary |
| 🧠 **MemoryAgent** | ChromaDB semantic storage for long-term experiment memory |

---

## Architecture

```
USER
 │
 ▼  Streamlit Dashboard (8 pages, dark mode, interactive charts)
 │
 ▼  FastAPI Backend (versioned API, Pydantic schemas, async)
 │
 ▼  Celery + Redis (non-blocking async task queue)
 │
 ▼  LangGraph Orchestrator (StateGraph, conditional routing, reflection loops)
 │
 ▼  Multi-Agent Pipeline (11 agents, shared AgentState TypedDict)
 │         ├── RAG Layer (sentence-transformers + ChromaDB)
 │         └── ML Layer (scikit-learn, XGBoost, LightGBM, Optuna)
 │
 ▼  Storage (SQLite checkpoints, Parquet, MLflow, ChromaDB, Reports)
```

---

## Features

- ✅ **13 ML models** with Optuna Bayesian HPO
- ✅ **SHAP + LIME** explainability for every prediction
- ✅ **Autonomous reflection loops** — retries when confidence drops
- ✅ **RAG experiment memory** — learns from every past run
- ✅ **Conversational assistant** — chat with your experiments
- ✅ **PDF + Markdown + JSON** reports with LLM executive summaries
- ✅ **MLflow tracking** — full experiment lineage
- ✅ **100% free** — local Ollama inference, no paid APIs
- ✅ **Docker Compose** — one command to run everything
- ✅ **GitHub Actions CI/CD** — automated testing + Docker builds

---
## 📖 Project Documentation

👉 **[View Full Project Documentation →](https://raghav-890.github.io)**

## Quick Start

### Option 1: Docker Compose (Easiest)

```bash
git clone https://github.com/your-username/autonomds.git
cd autonomds
cp .env.example .env
docker-compose up -d

# API:      http://localhost:8000/docs
# Frontend: http://localhost:8501
```

### Option 2: Local Development

```bash
# Prerequisites: Python 3.11+, Redis (optional), Ollama (optional)
git clone https://github.com/your-username/autonomds.git
cd autonomds

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
make dirs

# Terminal 1: API
uvicorn app.api.main:app --reload --port 8000

# Terminal 2: Frontend
streamlit run app/frontend/streamlit_app.py

# Optional: Local LLM
ollama pull llama3.2
```

### Option 3: Streamlit Cloud (Frontend) + Render (API)

See the full [Deployment Guide](docs/deployment.md).

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | Streamlit 1.35 | Dark mode, 8 pages, interactive Plotly charts |
| API | FastAPI 0.111 | Async, versioned, OpenAPI docs |
| Orchestration | LangGraph | StateGraph, reflection loops, checkpointing |
| LLM | Ollama + HuggingFace | 100% local, zero cost |
| Embeddings | sentence-transformers | Local, CPU-friendly |
| Vector DB | ChromaDB | Embedded, no separate server |
| Task Queue | Celery + Redis | Async pipeline execution |
| ML | scikit-learn, XGBoost, LightGBM | CPU-compatible |
| HPO | Optuna | Bayesian optimisation |
| Explainability | SHAP + LIME | TreeExplainer + KernelExplainer |
| Tracking | MLflow | Local file-based tracking |
| Deployment | Docker Compose | Production-ready |
| CI/CD | GitHub Actions | Lint, test, Docker build |

---

## Project Structure

```
autonomds/
├── app/
│   ├── agents/          # 11 specialised ML agents
│   │   ├── base_agent.py
│   │   ├── eda_agent.py
│   │   ├── training_agent.py
│   │   └── ...
│   ├── api/             # FastAPI backend
│   │   ├── main.py      # App + Celery
│   │   └── routes/      # upload, pipeline, experiments, reports
│   ├── frontend/        # Streamlit UI
│   │   ├── streamlit_app.py
│   │   ├── pages/       # 8 pages
│   │   └── styles/      # Dark mode CSS
│   ├── orchestration/   # LangGraph
│   │   ├── graph.py     # StateGraph definition
│   │   └── state.py     # AgentState TypedDict
│   ├── memory/          # ChromaDB + embeddings
│   ├── rag/             # RAG retrieval pipeline
│   ├── monitoring/      # MLflow tracker
│   └── utils/           # Config, logger, helpers, validators
├── tests/
│   ├── unit/            # 6 unit test files
│   └── integration/     # API + pipeline integration tests
├── docs/                # Architecture, agent design, RAG, deployment
├── docker/              # Dockerfiles (API, Frontend, Worker)
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check |
| `POST` | `/api/v1/upload` | Upload dataset file |
| `POST` | `/api/v1/pipeline/run` | Trigger ML pipeline |
| `GET` | `/api/v1/pipeline/status/{id}` | Poll pipeline progress |
| `GET` | `/api/v1/pipeline/result/{id}` | Get full results |
| `POST` | `/api/v1/pipeline/chat` | Chat with experiment assistant |
| `GET` | `/api/v1/experiments` | List all experiments |
| `GET` | `/api/v1/reports/{id}/pdf` | Download PDF report |
| `GET` | `/api/v1/reports/{id}/markdown` | Download Markdown report |

Interactive docs: `http://localhost:8000/docs`

---

## Development Commands

```bash
make help          # Show all available commands
make test          # Run full test suite with coverage
make test-unit     # Fast unit tests only
make lint          # Ruff linting
make format        # Black formatting
make typecheck     # mypy type checking
make docker-up     # Start Docker stack
make docker-down   # Stop Docker stack
make clean         # Remove caches and build artifacts
```

---

## Roadmap

- [ ] Kaggle dataset integration
- [ ] Time-series forecasting support (Prophet, LSTM)
- [ ] AutoML with Neural Architecture Search
- [ ] Multi-user experiment sharing
- [ ] Slack/Discord notifications on pipeline completion
- [ ] Export to Jupyter notebook
- [ ] GPU-accelerated training (CUDA support)
- [ ] OpenAI API integration (optional paid tier)
- [ ] REST API SDK (Python client library)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-agent`
3. Write tests for your changes
4. Run `make quality` to check lint + types
5. Submit a pull request

All contributions welcome — new agents, new ML models, UI improvements.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ as a production-grade portfolio project.<br>
<strong>Stars ⭐ are appreciated if this helped you!</strong>
</div>
