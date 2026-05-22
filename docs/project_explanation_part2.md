# AutonomDS — Complete Project Explanation (Part 2)
# Agent Deep-Dives, Data Flow, State Management

---

## The 11 Agents — What Each One Does

### Agent 1: DataIngestionAgent
**File:** `app/agents/ingestion_agent.py`

**Job:** Load the uploaded file, detect the schema, infer target column and task type.

**What it does step by step:**
1. Reads the file path from `state["raw_data_path"]`
2. Detects the file extension (.csv, .xlsx, .parquet, .db)
3. Loads it into a pandas DataFrame using the right reader
4. Computes statistics: n_rows, n_cols, column names, dtypes, memory usage
5. Calculates a SHA-256 checksum of the file
6. Infers which column is the target (looks for columns named "target", "label", "y", "output", etc.)
7. Infers task type: if target has 2 unique values → binary_classification; if ≤20 → multiclass; if continuous → regression
8. Saves dataset as Parquet for fast downstream reading
9. Writes all findings into AgentState

**Why Parquet for intermediate storage?**
- 5-10x faster to read than CSV for large datasets
- Preserves dtypes (CSV loses int/float distinctions)
- Compressed — uses less disk space

---

### Agent 2: EDAAgent
**File:** `app/agents/eda_agent.py`

**Job:** Automated Exploratory Data Analysis — understand the data before touching it.

**What it does step by step:**
1. Loads the raw file
2. **Missing values**: counts nulls per column, calculates % missing
3. **Correlation matrix**: Pearson correlation between all numeric columns
4. **Outlier detection**: IQR method — values below Q1-1.5×IQR or above Q3+1.5×IQR are flagged
5. **Class balance**: for classification targets, measures imbalance ratio (max_class/min_class)
6. **Skewness**: flags features with |skew| > 0.5
7. **Charts**: generates Plotly HTML files for each analysis (saved to `experiments/{exp_id}/eda_charts/`)
8. **LLM insights**: sends a statistical summary to the LLM and asks for 5 actionable insights in JSON format
9. **Fallback**: if LLM is offline, generates rule-based insights instead (no quality loss)

**Why IQR for outlier detection?**
- Robust to non-normal distributions
- Doesn't assume the data follows a bell curve
- Simple, fast, interpretable
- Alternative (Z-score) fails on skewed data

**Why save charts as HTML files?**
- Plotly HTML is interactive — users can zoom, hover, pan in the browser
- Streamlit reads them with `st.components.v1.html()`
- Avoids embedding base64 image strings in JSON (would be huge)

---

### Agent 3: DataCleaningAgent
**File:** `app/agents/cleaning_agent.py`

**Job:** Fix data quality issues found by the EDA agent.

**What it does step by step:**
1. Reads from `state["processed_data_path"]`
2. **Removes duplicates**: `df.drop_duplicates()`
3. **Drops high-missing columns**: any column with >60% missing is dropped
4. **Imputes numeric columns**: KNN imputation for small datasets (<5000 rows), median for larger ones
5. **Imputes categorical columns**: most frequent value
6. **Encodes categoricals**:
   - 2 unique values → LabelEncoder (0/1)
   - 3-10 unique values → One-Hot Encoding (pd.get_dummies)
   - >10 unique values → LabelEncoder (high cardinality)
7. **Encodes target** for classification: LabelEncoder if not already numeric
8. **Scales numeric features**: RobustScaler (resistant to outliers unlike StandardScaler)
9. **Leakage detection**: if any feature correlates >0.99 with target, it's dropped
10. Saves cleaned data as new Parquet file
11. Records all actions taken in `state["cleaning_actions"]`

**Why RobustScaler over StandardScaler?**
- StandardScaler: uses mean and std — both are heavily distorted by outliers
- RobustScaler: uses median and IQR — unaffected by extreme values
- EDA already flagged outliers exist → RobustScaler is the right choice

**Why KNN imputation for small datasets?**
- KNN looks at similar rows to fill missing values — more accurate than simple median
- But it's O(n²) — too slow for large datasets
- For large datasets, median imputation is fast and good enough

---

### Agent 4: FeatureEngineeringAgent
**File:** `app/agents/feature_engineering_agent.py`

**Job:** Select the most important features and optionally create new ones.

**What it does:**
1. **RFECV** (Recursive Feature Elimination with Cross-Validation): trains a quick estimator and recursively removes the weakest features until CV score stops improving
2. **Variance threshold**: removes features with near-zero variance (they carry no information)
3. Optionally adds **polynomial features** for regression tasks
4. Records which features were kept vs removed
5. Updates `state["selected_features"]` and `state["n_features_selected"]`

**Why RFECV over manual selection?**
- Automated — no human judgment needed
- Uses actual model performance as the selection criterion
- Cross-validated — avoids selecting features that overfit

---

### Agent 5: ModelSelectionAgent
**File:** `app/agents/model_selection_agent.py`

**Job:** Recommend which ML models to train.

**What it does:**
1. Checks `state["task_type"]` to know if it's classification or regression
2. Queries ChromaDB for similar past experiments via RAG
3. Runs quick baseline tests (5-fold CV, 20 trials) on 3-4 candidate models
4. Ranks them by baseline score
5. Selects the top 2-3 models to train fully
6. Writes `state["selected_model_types"]` = e.g., `["XGBClassifier", "RandomForestClassifier"]`

**Why not train all models?**
- Training 13 models with Optuna HPO would take 30-60 minutes
- Selecting the top performers for full training saves 70% of compute time
- The baseline quick-train reveals which families work for this data

---

### Agent 6: TrainingAgent
**File:** `app/agents/training_agent.py`

**Job:** Train the selected models with hyperparameter optimisation.

**What it does for each model:**
1. Defines a parameter search space (e.g., `n_estimators` 50-300, `max_depth` 3-15)
2. Runs **Optuna** with N trials — each trial trains the model with different params and measures CV score
3. Optuna uses **Tree-structured Parzen Estimator (TPE)** — a Bayesian method that learns which param regions are promising
4. Takes the best params, trains the final model on the full training set
5. Runs **k-fold cross-validation** to get reliable score estimate
6. Saves the model as `.joblib` file
7. Logs params, metrics, and the model file to MLflow
8. Records result in `state["trained_models"]`

**Why joblib over pickle?**
- joblib is specifically optimised for NumPy arrays (which sklearn models contain)
- Faster serialisation, better compression for large arrays

**Why cross-validation instead of a simple train/test split?**
- CV uses all data for both training and validation (just in different folds)
- Single split variance: depends on which rows happened to be in the test set
- CV gives a more reliable estimate of how the model will perform on unseen data

---

### Agent 7: EvaluationAgent
**File:** `app/agents/evaluation_agent.py`

**Job:** Rigorously evaluate all trained models and build a leaderboard.

**For classification tasks:**
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix
- Per-class metrics for multiclass

**For regression tasks:**
- RMSE, MAE, R², MAPE

**What it produces:**
- `state["leaderboard"]` — sorted list of all models by primary metric
- `state["best_model_name"]` — the winner
- `state["confusion_matrix"]` — for the best model
- `state["roc_auc"]` — for binary classification
- Sets `state["should_reflect"] = True` if best score < threshold (triggers reflection loop)

---

### Agent 8: ReflectionAgent (node_reflect in graph.py)
**File:** `app/orchestration/graph.py` (the `node_reflect` function)

**Job:** Review poor results and decide whether to retry training.

**How it works:**
1. Triggered when `state["should_reflect"] == True`
2. Sends `state["errors"]` and `state["reflection_notes"]` to the LLM
3. LLM responds with JSON: `{"action": "retry", "stage": "train", "reason": "..."}`
4. Increments `state["retry_count"]`
5. Router `route_after_reflect()` checks retry count — if <3, routes back to training

**Why a reflection loop?**
- Mimics how a human data scientist would respond to bad results
- "The model scored only 0.52 AUC — let me try different models or more HPO trials"
- Adds resilience — the system self-corrects instead of silently producing poor results

---

### Agent 9: ExplainabilityAgent
**File:** `app/agents/explainability_agent.py`

**Job:** Explain why the model makes its predictions.

**What it does:**
1. Loads the best model from `state["best_model_path"]`
2. Uses **SHAP TreeExplainer** for tree-based models (XGBoost, RF, LightGBM) — fast, exact
3. Falls back to **SHAP KernelExplainer** for other model types — slower, model-agnostic
4. Computes feature importance scores for every feature
5. Generates a SHAP summary plot (saved as HTML)
6. Writes `state["feature_importance"]` dict

**SHAP vs simple feature importance:**
- Random Forest's built-in feature importance is biased toward high-cardinality features
- SHAP values are mathematically grounded — based on Shapley values from cooperative game theory
- SHAP shows the direction of influence (positive/negative), not just magnitude

---

### Agent 10: ReportAgent
**File:** `app/agents/report_agent.py`

**Job:** Generate human-readable reports of the entire experiment.

**What it generates:**
1. **Markdown report**: structured summary with tables, metrics, insights
2. **PDF report**: rendered from the Markdown (using reportlab or markdown2pdf)
3. **JSON report**: machine-readable version of all results

**LLM-generated executive summary:**
- Sends all results to the LLM
- Asks for a 3-paragraph executive summary for non-technical stakeholders
- This is one of the highest-value LLM uses — translating numbers into business language

---

### Agent 11: MemoryAgent
**File:** `app/agents/memory_agent.py`

**Job:** Store this experiment's results in long-term memory for future RAG retrieval.

**What it stores:**
```
Document: "Dataset: sales_data.csv (50000 rows, 12 cols).
Task: regression. Best model: XGBRegressor. R²: 0.87.
Key features: revenue, cost, discount. Preprocessing: KNN imputation, RobustScaler."
```

**Why this matters:**
- Next time a similar dataset is uploaded, ModelSelectionAgent queries ChromaDB
- "Find past experiments similar to this regression task" → retrieves this document
- Agent uses past results to make better decisions: "Last time XGBoost worked great for regression on financial data — try it first"

---

## The Central State: AgentState

**File:** `app/orchestration/state.py`

This is the most important data structure in the entire system. It's a Python `TypedDict` — a dictionary with type annotations.

```python
class AgentState(TypedDict, total=False):
    experiment_id: str          # "exp-20240519-abc123"
    raw_data_path: str          # "/uploads/exp-xxx/data.csv"
    processed_data_path: str    # "/uploads/exp-xxx/data_cleaned.parquet"
    target_column: str          # "price"
    task_type: str              # "regression"
    eda_results: dict           # all EDA stats
    eda_charts: list[str]       # paths to HTML chart files
    trained_models: list[dict]  # all model results
    best_model_name: str        # "XGBRegressor"
    leaderboard: list[dict]     # ranked model comparison
    errors: list[str]           # any errors encountered
    should_reflect: bool        # trigger reflection loop?
    retry_count: int            # how many reflection retries so far
    ...
```

**Why TypedDict (total=False)?**
- `total=False` means every key is optional — agents add keys as they run
- Ingestion agent doesn't know trained_models yet — it just sets raw_data_path
- This lets agents incrementally populate the state without requiring all keys upfront
- LangGraph requires a TypedDict for its StateGraph

**Why one shared dict instead of separate objects?**
- All agents read from and write to the same dict — automatic information sharing
- LangGraph can checkpoint this dict to SQLite — resume from any point if it crashes
- Easy to serialize to JSON for API responses

---

## The LangGraph Pipeline

**File:** `app/orchestration/graph.py`

This is the wiring diagram of the entire system.

```
START → ingest → [route_after_ingest]
                    ├── "end" → END  (if file load failed)
                    └── "eda" → eda → clean → feature_eng → model_select → train → evaluate
                                                                                         │
                                                                            [route_after_evaluate]
                                                                            ├── "reflect" → reflect
                                                                            │                  │
                                                                            │         [route_after_reflect]
                                                                            │         ├── "train" → (back to train)
                                                                            │         └── "end" → END
                                                                            ├── "explain" → explain → report → memory → END
                                                                            └── "end" → END
```

**Key routing functions:**

`route_after_ingest(state)`:
- If file load failed (no dataset_info) → go to END
- Otherwise → go to EDA

`route_after_evaluate(state)`:
- If model score is poor (should_reflect=True) and retry_count < 2 → go to REFLECT
- If no leaderboard → go to END
- Otherwise → go to EXPLAIN

`route_after_reflect(state)`:
- If retry_count >= 3 → go to END (stop infinite loops)
- Otherwise → go back to TRAIN

**Why LangGraph's StateGraph over a simple for-loop?**
- Conditional routing: can branch, loop, retry — impossible with a linear loop
- Checkpointing: if the server crashes mid-pipeline, it can resume from the last node
- Each node is a pure function (state in → state out) — easy to test in isolation
- Visualisable: `graph.get_graph().draw_mermaid()` produces a diagram

---

## The RAG System

**Files:** `app/rag/retrieval.py`, `app/memory/experiment_memory.py`, `app/memory/chroma_store.py`

RAG = Retrieval-Augmented Generation. Instead of asking the LLM a question cold, we first retrieve relevant past information and include it in the prompt.

**Flow:**
```
New experiment starts
        │
        ▼
RAGRetriever.retrieve_for_model_selection(dataset_description, task_type)
        │
        ▼
Converts description to 384-dim vector (sentence-transformers)
        │
        ▼
ChromaDB cosine similarity search → returns top-5 most similar past experiments
        │
        ▼
Formats results as text context
        │
        ▼
ModelSelectionAgent includes this in its LLM prompt:
"Here are similar past experiments: [context]
Given this, which models would you recommend?"
        │
        ▼
LLM makes a more informed recommendation
```

**Three RAG retrieval modes:**
1. `retrieve_for_model_selection()` — "what models worked for similar datasets?"
2. `retrieve_for_preprocessing()` — "what cleaning steps worked before?"
3. `retrieve_for_chat()` — conversational assistant queries

**Why cosine similarity?**
- Measures the angle between two vectors — unaffected by vector length
- "Long document about XGBoost regression" and "Short note: XGBoost regression" map to nearby vectors
- Euclidean distance would penalise the longer document unfairly

---

## The BaseAgent Pattern

**File:** `app/agents/base_agent.py`

Every one of the 11 agents inherits from `BaseAgent`. This enforces a consistent interface:

```python
class SomeAgent(BaseAgent):
    name = "some_agent"
    stage = PipelineStage.TRAINING

    def execute(self, state: AgentState) -> AgentState:
        # This is the only method each agent MUST implement
        # All the surrounding lifecycle is handled by BaseAgent.run()
        ...
```

When `agent.run(state)` is called, BaseAgent automatically:
1. Creates an `AgentExecutionRecord` (tracks start time, status)
2. Updates `state["current_agent"]` and `state["current_stage"]`
3. Calls `self.execute(state)` — the agent's actual logic
4. Calls `self.compute_confidence(state)` — 0.0 to 1.0 score
5. If confidence < 0.6 → sets `state["should_reflect"] = True`
6. Appends a success/failure message to `state["messages"]`
7. Records duration in the execution record
8. Handles retries with exponential backoff (2^attempt seconds between retries)

**Why this pattern?**
- DRY (Don't Repeat Yourself): logging, retry, tracking logic written once
- Every agent is guaranteed to produce the same metadata format
- Easy to add new agents — just implement `execute()`
- Consistent error handling across all 11 agents

---

## The Configuration System

**File:** `app/utils/config.py`

All settings come from environment variables. The `AppSettings` class uses Pydantic BaseSettings:

```python
class AppSettings(BaseSettings):
    llm_provider: LLMProvider = Field(default=LLMProvider.AUTO)
    ollama_base_url: str = Field(default="http://localhost:11434")
    huggingface_api_token: Optional[str] = Field(default=None)
    optuna_n_trials: int = Field(default=30, gt=0)
    ...
```

**Smart LLM provider resolution:**
```python
@property
def effective_llm_provider(self) -> LLMProvider:
    if self.llm_provider == "auto":
        try:
            httpx.get(f"{self.ollama_base_url}/api/tags", timeout=3.0)
            return LLMProvider.OLLAMA   # Ollama is running locally
        except:
            return LLMProvider.HUGGINGFACE  # Fall back to cloud
```

Set `LLM_PROVIDER=auto` and the system picks the right LLM automatically.

**Why environment variables instead of a config file?**
- 12-Factor App methodology: config belongs in the environment, not in code
- Different environments (dev/prod/test) just set different env vars — no file edits
- Never accidentally commit secrets to git
- Docker/Kubernetes/Render all support env vars natively

---

## The API Design

**File:** `app/api/main.py` + `app/api/routes/`

**4 route groups:**

| Router | Prefix | Endpoints |
|--------|--------|-----------|
| upload | /api/v1 | POST /upload |
| pipeline | /api/v1 | POST /pipeline/run, GET /pipeline/status/{id}, GET /pipeline/result/{id}, POST /pipeline/chat |
| experiments | /api/v1 | GET /experiments, GET /experiments/{id} |
| reports | /api/v1 | GET /reports/{id}/pdf, GET /reports/{id}/markdown, GET /reports/{id}/json |

**The upload → pipeline flow:**
```
POST /api/v1/upload  (multipart/form-data with file)
    → FileValidator checks extension, size, reads into DataFrame
    → Generates experiment_id ("exp-20240519-a3f9bc")
    → Saves file to uploads/{exp_id}/
    → Returns: {experiment_id, filename, n_rows, n_cols, detected_target, ...}

POST /api/v1/pipeline/run  (body: {experiment_id, target_column, ...})
    → Builds initial AgentState dict
    → Sends to Celery: run_pipeline_task.delay(state)
    → Returns immediately: {task_id, status: "queued"}

GET /api/v1/pipeline/status/{task_id}
    → Checks Celery result backend (Redis)
    → Returns: {status: "running" | "completed" | "failed", progress: 0.7}

GET /api/v1/pipeline/result/{task_id}
    → Returns full AgentState as JSON
```

**Why version prefix `/api/v1/`?**
- Allows breaking changes in v2 without breaking existing v1 clients
- API clients just specify which version they use

---

## Testing Strategy

**58 tests across 8 files — 100% passing.**

**Unit tests** (`tests/unit/`):
- Each agent tested in isolation
- LLM calls mocked: `agent.ask_llm = MagicMock(return_value="...")`
- Uses real `tmp_path` temp directories — not mocked file I/O
- No network calls, no Redis, no Ollama needed

**Integration tests** (`tests/integration/`):
- `test_pipeline.py`: tests the LangGraph graph compiles, all 11 nodes present, routing functions correct
- `test_api.py`: uses `fastapi.testclient.TestClient` — real HTTP calls to the app without a live server

**Key design decision — mock the LLM, not the data:**
```python
agent.ask_llm = MagicMock(return_value='["insight 1", "insight 2"]')
```
- We test that the agent correctly uses the LLM response
- We don't test the LLM itself (that's the LLM provider's job)
- Tests run in seconds, not minutes

**MLflow patched via sys.modules:**
```python
with patch.dict(sys.modules, {"mlflow": FakeMLflow()}):
    agent.run(state)
```
- The training agent does `import mlflow` inside a method
- Can't patch a module-level attribute that doesn't exist
- Patching sys.modules makes Python find the fake version first

---

## File Structure Explained

```
autonomous-ds-agent/
├── app/
│   ├── agents/           # 11 agents + BaseAgent
│   ├── api/
│   │   ├── main.py       # FastAPI app + Celery setup
│   │   ├── routes/       # upload.py, pipeline.py, experiments.py, reports.py
│   │   └── schemas.py    # Pydantic request/response models
│   ├── frontend/
│   │   ├── streamlit_app.py   # Entry point, navigation
│   │   ├── pages/             # 8 pages (home, upload, eda, pipeline, etc.)
│   │   └── styles/theme.py    # Injected CSS for dark mode
│   ├── memory/
│   │   ├── chroma_store.py    # ChromaDB wrapper
│   │   ├── embeddings.py      # sentence-transformers wrapper
│   │   └── experiment_memory.py  # High-level interface
│   ├── monitoring/
│   │   └── mlflow_tracker.py  # MLflow with graceful degradation
│   ├── orchestration/
│   │   ├── graph.py      # LangGraph StateGraph definition
│   │   └── state.py      # AgentState TypedDict
│   ├── rag/
│   │   └── retrieval.py  # 3 retrieval modes
│   ├── tools/
│   │   └── data_tools.py # Shared utilities
│   └── utils/
│       ├── config.py     # Pydantic settings (env vars)
│       ├── helpers.py    # now_iso(), generate_experiment_id(), etc.
│       ├── logger.py     # structlog setup
│       └── validators.py # File format/size validation
├── tests/
│   ├── unit/             # Per-agent unit tests
│   └── integration/      # API + pipeline integration tests
├── docker/
│   ├── Dockerfile.api    # API container
│   ├── Dockerfile.frontend  # Streamlit container
│   └── Dockerfile.worker # Celery worker container
├── docs/                 # 4 documentation files
├── docker-compose.yml    # Full stack definition
├── Makefile              # Developer shortcuts
├── pyproject.toml        # Project config + pytest settings
└── requirements.txt      # All dependencies
```

---

## How a Real Request Flows End-to-End

```
1. User opens Streamlit at localhost:8501
2. User clicks "Upload Dataset", selects "titanic.csv"

3. Streamlit POST /api/v1/upload
   → FileValidator reads CSV, checks size, extension
   → Generates exp_id = "exp-20240519-a3f9bc"
   → Saves file to uploads/exp-20240519-a3f9bc/titanic.csv
   → Returns {experiment_id, n_rows: 891, detected_target: "Survived"}

4. User confirms target="Survived", clicks "Run Pipeline"

5. Streamlit POST /api/v1/pipeline/run
   → Builds AgentState with experiment_id, raw_data_path, target_column="Survived"
   → celery_app.send_task("run_pipeline_task", args=[state])
   → Returns {task_id: "celery-uuid-xxx", status: "queued"}

6. Streamlit polls GET /api/v1/pipeline/status/celery-uuid-xxx every 3 seconds

7. Meanwhile in Celery worker:
   pipeline_graph.invoke(state) calls:

   node_ingest(state):
     - Loads titanic.csv into DataFrame
     - Detects: 891 rows, 12 cols, target="Survived", task="binary_classification"
     - Saves as titanic.parquet
     - Sets state["dataset_info"], state["processed_data_path"]

   node_eda(state):
     - Finds: Age has 177 missing (19.9%), Cabin has 687 missing (77%)
     - Plots missing values bar chart → missing_values.html
     - Plots correlation heatmap → correlation_matrix.html
     - LLM generates 5 insights about the Titanic dataset
     - Sets state["eda_results"], state["eda_charts"], state["eda_warnings"]

   node_clean(state):
     - Drops Cabin (77% missing)
     - KNN imputes Age (177 missing values)
     - Mode imputes Embarked (2 missing)
     - One-hot encodes Sex, Embarked
     - Label encodes Name, Ticket (high cardinality)
     - RobustScaler on Fare, Age, SibSp, Parch
     - Saves cleaned_titanic.parquet
     - Sets state["cleaning_actions"] = ["Dropped Cabin", "KNN imputed Age", ...]

   node_feature_eng(state):
     - RFECV selects 8 of 12 features
     - Removes PassengerId (near-zero variance with target)
     - Sets state["selected_features"]

   node_model_select(state):
     - Queries ChromaDB: "binary classification, tabular, ~900 rows"
     - Quick baseline: XGBoost 0.81, RandomForest 0.79, LogisticRegression 0.77
     - Sets state["selected_model_types"] = ["XGBClassifier", "RandomForestClassifier"]

   node_train(state):
     - XGBClassifier: Optuna 15 trials → best params → CV ROC-AUC 0.87
     - RandomForestClassifier: Optuna 15 trials → CV ROC-AUC 0.84
     - Saves both as .joblib files
     - Logs to MLflow
     - Sets state["trained_models"], state["best_model_name"] = "XGBClassifier"

   node_evaluate(state):
     - XGBoost: AUC 0.87, Accuracy 0.82, F1 0.81
     - RandomForest: AUC 0.84, Accuracy 0.80, F1 0.79
     - Confusion matrix for XGBoost
     - Sets state["leaderboard"], state["should_reflect"] = False (good results)

   node_explain(state):
     - SHAP TreeExplainer on XGBoost model
     - Feature importance: Sex_male(0.31), Fare(0.18), Age(0.15), Pclass(0.14)...
     - Saves SHAP summary plot
     - Sets state["feature_importance"]

   node_report(state):
     - LLM writes executive summary: "The XGBoost model achieved 87% AUC on Titanic survival..."
     - Generates Markdown report with all metrics, charts, feature importance
     - Generates PDF from Markdown
     - Sets state["pdf_report_path"], state["markdown_report_path"]

   node_memory(state):
     - Creates document string summarising this experiment
     - Embeds with sentence-transformers
     - Stores in ChromaDB
     - Sets state["memory_stored"] = True

8. Pipeline returns final state dict

9. Celery stores result in Redis

10. Streamlit receives status="completed"
    → Loads EDA Explorer page with all charts and insights
    → Loads Model Leaderboard with AUC scores
    → Loads report download buttons
```

Total time: approximately 3-8 minutes depending on dataset size and hardware.

---

## Summary

AutonomDS is a production-grade system that demonstrates:

- **Multi-agent AI architecture** using LangGraph StateGraph
- **Clean separation of concerns** — each agent owns exactly one stage
- **Graceful degradation** — works without Ollama, without MLflow, with rule-based fallbacks
- **Full observability** — structured logging, MLflow tracking, execution records
- **Type safety** — Pydantic everywhere: config, API schemas, agent state
- **Testable design** — 58 unit + integration tests, 100% passing
- **Production readiness** — Docker, CI/CD, cloud deployment support
- **Zero cost** — Ollama for local LLM, ChromaDB embedded, SQLite for persistence
