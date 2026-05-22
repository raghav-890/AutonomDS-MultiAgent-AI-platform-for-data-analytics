# AutonomDS — System Architecture

## Overview

AutonomDS is a production-grade autonomous multi-agent data science platform. It orchestrates a team of 11 specialised AI agents through a directed graph (LangGraph) to take a raw dataset from upload to trained, explained, and reported ML model — autonomously.

## Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  STREAMLIT FRONTEND                      │
│  Upload │ EDA │ Pipeline │ Models │ Experiments │ Chat  │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP (REST)
┌─────────────────▼───────────────────────────────────────┐
│                  FASTAPI BACKEND                         │
│  /upload  /pipeline/run  /pipeline/status  /chat        │
│  Pydantic validation │ CORS │ Exception handling         │
└─────────────────┬───────────────────────────────────────┘
                  │ Celery task.delay()
┌─────────────────▼───────────────────────────────────────┐
│              CELERY + REDIS BROKER                       │
│  Non-blocking task queue │ Pipeline worker pool          │
└─────────────────┬───────────────────────────────────────┘
                  │ pipeline_graph.invoke(state)
┌─────────────────▼───────────────────────────────────────┐
│              LANGGRAPH ORCHESTRATOR                      │
│  StateGraph │ Conditional routing │ Reflection loops     │
│  SQLite checkpointing │ Resumability                     │
└─────────────────┬───────────────────────────────────────┘
                  │ AgentState TypedDict
┌─────────────────▼───────────────────────────────────────┐
│              MULTI-AGENT PIPELINE                        │
│                                                          │
│  DataIngestionAgent → EDAAgent → DataCleaningAgent      │
│    → FeatureEngineeringAgent → ModelSelectionAgent       │
│      → TrainingAgent → EvaluationAgent                  │
│        → [ReflectionAgent?] → ExplainabilityAgent        │
│          → ReportAgent → MemoryAgent                     │
└────────────────┬────────────────┬────────────────────────┘
                 │                │
┌────────────────▼──────┐  ┌─────▼────────────────────────┐
│  RAG + MEMORY LAYER   │  │  ML PIPELINE LAYER           │
│  sentence-transformers│  │  scikit-learn │ XGBoost       │
│  ChromaDB             │  │  LightGBM │ Optuna HPO        │
│  ExperimentMemory     │  │  SHAP │ LIME                  │
└────────────────┬──────┘  └─────┬────────────────────────┘
                 │                │
┌────────────────▼────────────────▼────────────────────────┐
│              STORAGE LAYER                               │
│  SQLite (pipeline state) │ Parquet (processed data)      │
│  MLflow (experiment tracking) │ ChromaDB (vectors)       │
│  Filesystem (models, charts, reports)                    │
└──────────────────────────────────────────────────────────┘
```

## Technology Choices

| Layer | Technology | Reason |
|---|---|---|
| Frontend | Streamlit | Rapid data app UI, Python-native, free hosting |
| API | FastAPI | Modern async Python API, automatic OpenAPI docs |
| Orchestration | LangGraph | Stateful graph-based agent orchestration, built for LLM pipelines |
| LLM | Ollama + HuggingFace | 100% local inference, zero cost, privacy-preserving |
| Embeddings | sentence-transformers | Local, high-quality, CPU-friendly |
| Vector DB | ChromaDB | Embedded (no separate server), open-source |
| Task Queue | Celery + Redis | Production-proven async task execution |
| Experiment Tracking | MLflow | Free, self-hosted, industry standard |
| ML | scikit-learn, XGBoost, LightGBM | Industry-proven, CPU-compatible |
| HPO | Optuna | Bayesian optimisation, excellent scikit-learn integration |
| Explainability | SHAP + LIME | Two complementary explainability methods |

## State Flow

All agents communicate through a single **AgentState TypedDict**. No direct agent-to-agent calls.

```
Initial State (experiment_id, raw_data_path)
    → DataIngestionAgent adds: dataset_info, target_column, task_type
    → EDAAgent adds: eda_results, eda_charts, eda_insights
    → DataCleaningAgent adds: processed_data_path, cleaning_report
    → FeatureEngineeringAgent adds: selected_features, feature_report
    → ModelSelectionAgent adds: candidate_models
    → TrainingAgent adds: trained_models, best_model_name
    → EvaluationAgent adds: leaderboard, evaluation_results
    → [ReflectionAgent adds: reflection_notes if confidence < 0.6]
    → ExplainabilityAgent adds: shap_values_path, feature_importance
    → ReportAgent adds: pdf_report_path, markdown_report_path
    → MemoryAgent adds: similar_experiments, memory_stored
    → Final State (pipeline_complete=True)
```

## Scalability Design

- **Horizontal scaling**: Celery workers can be scaled independently across machines
- **Stateless API**: FastAPI is stateless; state lives in Redis + SQLite
- **Modular agents**: Each agent is a self-contained Python class — swap implementations freely
- **Pluggable LLM**: Switching from Ollama → OpenAI requires changing one setting value
- **Pluggable vector DB**: ChromaDB → Pinecone → Weaviate with one class change

## Security

- File validation (extension whitelist, size limits, checksum)
- No code execution from user input (RestrictedPython sandboxing available)
- Environment variable protection (no secrets in code)
- Temp files in isolated per-experiment directories
