# AutonomDS — Complete Project Explanation (Part 1)

## What Is AutonomDS?

AutonomDS is a **multi-agent AI platform** that automates the entire data science workflow. Instead of a data scientist manually writing code for EDA, cleaning, model selection, training, and reporting, AutonomDS has **11 specialised AI agents** that each own one step of that pipeline — and they coordinate through a central orchestrator.

Think of it like a data science company where each employee is an AI agent with a specific job.

---

## The Problem It Solves

A typical data science project involves:
1. Loading and inspecting data
2. Cleaning missing values, encoding categories, scaling
3. Selecting which features matter
4. Choosing the right ML model
5. Tuning hyperparameters
6. Evaluating results
7. Explaining predictions
8. Writing a report

This takes a human data scientist days. AutonomDS does it in minutes, autonomously.

---

## High-Level Architecture

```
User uploads CSV/Excel/Parquet/SQLite
         │
         ▼
   Streamlit Frontend  (browser UI)
         │  HTTP
         ▼
   FastAPI Backend     (REST API)
         │  Celery task
         ▼
   LangGraph Pipeline  (orchestrator)
         │
         ▼
   11 AI Agents run in sequence
         │
         ▼
   Results: trained model + PDF report + charts
```

Every layer has a specific job and they are completely separated from each other.

---

## Technology Choices & Why

### 1. Python 3.11
**Why:** Most ML/AI libraries are Python-first. Type hints in 3.11 are faster and better. Nearly all data science tools (pandas, sklearn, plotly) are Python.

**Alternatives rejected:**
- Julia: faster numerics but tiny ecosystem
- R: great for stats but terrible for web APIs and deployment
- Go: no ML library ecosystem

---

### 2. FastAPI (Backend API)
**Why:** 
- Async-native — handles concurrent requests without blocking
- Auto-generates interactive API docs at `/docs`
- Pydantic validation built in — every request/response is type-checked
- 2-3x faster than Flask in benchmarks

**Alternatives rejected:**
- **Flask**: Synchronous by default, no built-in validation, no auto docs
- **Django**: Too heavy, built for page-rendered apps not REST APIs
- **Express (Node)**: No Python ML ecosystem access

---

### 3. Streamlit (Frontend)
**Why:**
- Python-native — no JavaScript needed
- Deploys for free on Streamlit Community Cloud
- Built-in state management, file upload, interactive widgets
- Dark mode theming via injected CSS

**Alternatives rejected:**
- **React/Next.js**: Requires JavaScript, separate frontend deployment, much more complex
- **Gradio**: Good for demos but limited layout control
- **Dash**: More complex, Plotly-specific, harder theming

---

### 4. LangGraph (Orchestration)
**Why:**
- Designed specifically for multi-agent AI pipelines
- **StateGraph** lets agents share a typed state dict
- Supports **conditional routing** — pipeline can branch based on results
- Built-in **checkpointing** — pipeline can resume from any node if it crashes
- Made by LangChain team, actively maintained

**Alternatives rejected:**
- **Plain Python functions**: No state persistence, no retry, no conditional routing
- **LangChain LCEL**: Linear chains only, no graph structure, no cycles
- **Prefect/Airflow**: Built for data pipelines, not AI agents — no LLM integration, overkill for this
- **AutoGen**: Microsoft's multi-agent framework, but agents communicate via chat which is unstructured and hard to test

---

### 5. Celery + Redis (Task Queue)
**Why:**
- Running the ML pipeline can take 5-15 minutes
- Without a task queue, the HTTP request would time out
- Celery runs the pipeline in a **background worker process**
- The API immediately returns a task ID
- Frontend polls for status every few seconds
- Redis is the message broker that connects FastAPI ↔ Celery

**Alternatives rejected:**
- **FastAPI BackgroundTasks**: Single-process, no retry, no status tracking
- **RQ (Redis Queue)**: Simpler but fewer features, less production-ready
- **Kafka**: Massively over-engineered for this use case

---

### 6. LangChain (LLM Interface)
**Why:**
- Provides a unified interface for both Ollama and HuggingFace
- Same `ChatModel.invoke()` call works regardless of which LLM is running
- When you switch from Ollama to HuggingFace, zero code changes needed

**Alternatives rejected:**
- **Direct HTTP to Ollama API**: Loses portability — code breaks when switching LLMs
- **OpenAI SDK**: Paid API, not free

---

### 7. Ollama (Local LLM)
**Why:**
- Runs LLMs (Mistral, Llama3, DeepSeek) 100% locally
- Zero API cost, zero rate limits, works offline
- Privacy: your data never leaves your machine

**Alternatives rejected:**
- **OpenAI GPT-4**: Costs money, data leaves your system
- **Anthropic Claude**: Same issues
- **Hugging Face Transformers direct**: Requires GPU, complex setup

---

### 8. ChromaDB (Vector Database)
**Why:**
- Runs **embedded** — no separate server to manage
- Stores experiment summaries as vector embeddings
- Semantic search: "find past experiments similar to this one"
- Used for RAG (Retrieval-Augmented Generation) to inform agent decisions

**Alternatives rejected:**
- **Pinecone**: Cloud-only, costs money
- **Weaviate**: Requires a separate Docker container
- **FAISS**: Facebook's library, no persistence built-in, lower-level

---

### 9. Sentence-Transformers (Embeddings)
**Why:**
- Converts text (experiment summaries) into 384-dimensional vectors
- `all-MiniLM-L6-v2` model is ~80MB — fast on CPU, no GPU needed
- 100% local, no API calls

**Alternatives rejected:**
- **OpenAI text-embedding-ada-002**: Costs money per embedding
- **Cohere Embed**: Also paid

---

### 10. Scikit-learn + XGBoost + LightGBM (ML)
**Why:**
- Sklearn: industry standard, 13+ algorithms, consistent API
- XGBoost: best gradient boosting for tabular data
- LightGBM: faster than XGBoost, handles large datasets better
- All CPU-compatible — works without GPU

**Alternatives rejected:**
- **PyTorch/TensorFlow**: Overkill for tabular ML, requires GPU for practical use
- **H2O AutoML**: Great but heavy dependency, Java runtime needed

---

### 11. Optuna (Hyperparameter Optimisation)
**Why:**
- Bayesian optimisation — smarter than random search
- Parallel trial execution
- Pruning: stops bad trials early, saving compute time
- Works with any model via a simple `objective()` function

**Alternatives rejected:**
- **GridSearchCV**: Exhaustive search — exponentially slow as params grow
- **RandomSearchCV**: Faster but no intelligence — random guesses
- **Ray Tune**: More powerful but requires a Ray cluster

---

### 12. SHAP (Explainability)
**Why:**
- Mathematical guarantee of feature importance (Shapley values from game theory)
- Works with any model via TreeExplainer (fast) or KernelExplainer (model-agnostic)
- Industry-standard for ML explainability

**Alternatives rejected:**
- **LIME**: Approximations only, less mathematically rigorous
- **ELI5**: Limited model support, less maintained

---

### 13. MLflow (Experiment Tracking)
**Why:**
- Logs every model's parameters, metrics, artifacts
- Runs locally as SQLite — no server needed
- Provides a UI to compare experiments
- Industry standard at companies like Databricks, Microsoft

**Alternatives rejected:**
- **Weights & Biases**: Paid for team features, requires account
- **Neptune.ai**: Also paid
- **Manual logging**: No searchability, no comparison UI

---

### 14. Plotly (Charts)
**Why:**
- Interactive charts (hover, zoom, pan) — much better than static images
- Saves as HTML files — embedded directly in Streamlit
- Dark theme via `paper_bgcolor` and `plot_bgcolor`

**Alternatives rejected:**
- **Matplotlib**: Static images only, limited interactivity
- **Seaborn**: Beautiful but static, poor Streamlit integration
- **Bokeh**: Similar to Plotly but smaller ecosystem

---

### 15. Structlog (Logging)
**Why:**
- Structured JSON logs — every log entry is a dict, not a string
- Makes logs searchable in production (e.g., `grep` by `agent=eda_agent`)
- Context binding: add `app=AutonomDS` to every log automatically

**Alternatives rejected:**
- **Python logging**: Unstructured strings, hard to parse in production
- **Loguru**: Better than logging but not structured JSON by default

---

### 16. Pydantic + Pydantic-Settings (Config & Validation)
**Why:**
- All config comes from environment variables — 12-factor app compliance
- Type-checked at startup — if `OPTUNA_N_TRIALS=abc` is set, it fails immediately with a clear error
- `.env` file support built-in

**Alternatives rejected:**
- **python-decouple**: No type validation
- **dynaconf**: More complex, less widely used

---

### 17. Docker + Docker Compose (Containerisation)
**Why:**
- One command to run the entire stack: `docker-compose up`
- Eliminates "works on my machine" problems
- Each service (API, frontend, worker) runs in isolated containers

**Alternatives rejected:**
- **No containers**: Dependency conflicts between Python packages
- **Kubernetes**: Massively over-engineered for a single-server deployment

---

### 18. GitHub Actions (CI/CD)
**Why:**
- Free for public repositories (2,000 min/month for private)
- Runs tests automatically on every push
- Catches bugs before they reach production

**Alternatives rejected:**
- **Jenkins**: Requires a server to run on
- **CircleCI**: Free tier is limited

---

### 19. Pytest (Testing)
**Why:**
- Industry standard Python test framework
- `tmp_path` fixture auto-creates/cleans temp directories
- `pytest-mock` for clean mocking
- `pytest-cov` for coverage reports

**Alternatives rejected:**
- **unittest**: Verbose, less readable
- **nose**: Deprecated

---
