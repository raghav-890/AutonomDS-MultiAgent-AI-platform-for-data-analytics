# AutonomDS — Agent Design Reference

## Agent Architecture

Every agent in AutonomDS inherits from `BaseAgent` and follows the same lifecycle:

```
run(state)
  │
  ├── Record AgentExecutionRecord (started_at, status=RUNNING)
  ├── Update state: current_agent, current_stage
  │
  ├── _run_with_retry(state)
  │     └── execute(state)  ← concrete agent logic here
  │         Retry up to max_retries times on transient failure
  │
  ├── compute_confidence(state) → float [0.0, 1.0]
  │     If confidence < 0.6: set should_reflect=True
  │
  ├── Append NL message to state["messages"]
  ├── Update AgentExecutionRecord (completed_at, duration)
  └── Return updated AgentState
```

## Agent Registry

### 1. DataIngestionAgent
**Stage:** `ingestion`  
**State Reads:** `raw_data_path`, `target_column` (optional hint)  
**State Writes:** `dataset_info`, `target_column`, `task_type`, `feature_columns`, `numeric_columns`, `categorical_columns`  
**Confidence Logic:** 1.0 if dataset loaded + target identified, 0.5 if target guessed, 0.3 on error  

Supported formats: CSV, Excel (.xlsx), Parquet, SQLite databases.  
Infers target column by name matching (`target`, `label`, `class`, `y`, `output`), falls back to last column.  
Task type inferred from target dtype and cardinality.

### 2. EDAAgent
**Stage:** `eda`  
**State Reads:** `raw_data_path`, `target_column`  
**State Writes:** `eda_results`, `eda_charts`, `eda_insights`, `eda_warnings`  
**Confidence Logic:** 1.0 normally, 0.6 if ≥5 data quality warnings  

Generates Plotly HTML charts saved to `reports/{exp_id}/eda_charts/`:
- `missing_values.html` — bar chart of null percentages per column
- `correlation_matrix.html` — Pearson heatmap
- `class_balance.html` — bar + pie chart of target distribution
- `distributions.html` — histogram grid for numeric features

LLM asks for 5 actionable insights in JSON format. Falls back to rule-based insights if LLM unavailable.

### 3. DataCleaningAgent
**Stage:** `cleaning`  
**State Reads:** `raw_data_path`, `target_column`, `numeric_columns`, `categorical_columns`  
**State Writes:** `processed_data_path`, `cleaning_report`, `cleaning_actions`  
**Confidence Logic:** Penalised if >20% missing remains after cleaning  

Operations (in order):
1. Duplicate row removal
2. Numeric imputation (median for skewed, mean for normal)
3. Categorical imputation (mode)
4. Categorical encoding (OrdinalEncoder for tree models, OneHot for linear)
5. Numeric scaling (RobustScaler to handle outliers)
6. LLM leakage detection query

Saves processed data as Parquet for efficient loading by downstream agents.

### 4. FeatureEngineeringAgent
**Stage:** `feature_engineering`  
**State Reads:** `processed_data_path`, `target_column`, `task_type`  
**State Writes:** `selected_features`, `n_features_original`, `n_features_selected`, `feature_report`  
**Confidence Logic:** Based on variance explained / feature retention ratio  

Operations:
- RFECV (Recursive Feature Elimination w/ Cross-Validation) for feature selection
- Polynomial interaction features (degree=2) for small feature sets
- PCA dimensionality reduction if n_features > 50

### 5. ModelSelectionAgent
**Stage:** `model_selection`  
**State Reads:** `task_type`, `dataset_info`, `similar_experiments` (from RAG)  
**State Writes:** `candidate_models`, `selected_model_types`  
**Confidence Logic:** Always 1.0 (deterministic selection)  

Selection logic:
- Classification: `[LogisticRegression, RandomForest, XGBoost, LightGBM]`
- Regression: `[Ridge, RandomForest, XGBoost, LightGBM]`
- RAG context from similar past experiments can influence selection

### 6. TrainingAgent
**Stage:** `training`  
**State Reads:** `processed_data_path`, `target_column`, `task_type`, `selected_model_types`  
**State Writes:** `trained_models`, `best_model_name`, `best_model_path`  
**Confidence Logic:** Based on CV mean score  

For each candidate model:
1. Define Optuna search space
2. Run N trials of Bayesian HPO
3. 5-fold cross-validation with best params
4. Save model with joblib
5. Log to MLflow

### 7. EvaluationAgent
**Stage:** `evaluation`  
**State Reads:** `processed_data_path`, `trained_models`, `target_column`, `task_type`  
**State Writes:** `leaderboard`, `evaluation_results`, `confusion_matrix`, `roc_auc`  
**Confidence Logic:** Best metric score as proxy for confidence  

Generates:
- `leaderboard.html` — horizontal bar chart ranked by primary metric
- `confusion_matrix.html` — heatmap for classification tasks

### 8. ExplainabilityAgent
**Stage:** `explainability`  
**State Reads:** `best_model_path`, `processed_data_path`, `target_column`, `task_type`  
**State Writes:** `shap_values_path`, `feature_importance`, `explainability_report`  
**Confidence Logic:** 1.0 if SHAP succeeded, 0.7 if fallback to built-in importance  

SHAP approach:
- Tree models → `TreeExplainer` (fast, exact)
- Linear models → `LinearExplainer`
- Other → `KernelExplainer` (slower, model-agnostic)

### 9. ReportAgent
**Stage:** `report`  
**State Reads:** All previous agent outputs  
**State Writes:** `pdf_report_path`, `markdown_report_path`, `report_summary`  

Generates 3 artifacts:
1. **PDF** — ReportLab professional report with leaderboard table
2. **Markdown** — Full experiment report for GitHub/GitLab display
3. **JSON** — Machine-readable summary for API consumption
4. **LLM executive summary** — 3-paragraph business-readable summary

### 10. MemoryAgent
**Stage:** `memory`  
**State Reads:** All previous agent outputs  
**State Writes:** `memory_stored`, `similar_experiments`  

1. Builds an experiment document string from key metadata
2. Embeds with `sentence-transformers`
3. Stores in ChromaDB
4. Retrieves top-3 similar past experiments for the result display

### 11. ReflectionAgent (graph node, not a class)
**Trigger:** `should_reflect=True` (set when any agent has confidence < 0.6)  
**Max activations:** 2 per pipeline run (`retry_count < 2`)  
**Action:** LLM reviews errors and reflection_notes, decides `retry | continue`, routes back to `train`
