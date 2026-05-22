# AutonomDS — RAG System Documentation

## What This RAG Is (And What It Isn't)

### ❌ NOT a simple PDF chatbot RAG
Most RAG tutorials embed PDFs and answer questions about them. That is NOT what AutonomDS implements.

### ✅ Agentic Experiment Memory RAG
AutonomDS implements **experiment memory retrieval** — a form of RAG where:
- The "documents" are **past ML experiment results**
- Retrieval augments **agent decision-making** (not just user chat)
- The system gets **smarter with every experiment** it runs

## Architecture

```
New Dataset Uploaded
        ↓
Experiment runs → Results generated
        ↓
EmbeddingModel (sentence-transformers)
encodes experiment summary
        ↓
ChromaDB stores vector + metadata
        ↓
        ↓ (Next experiment)
        ↓
Query: "binary classification, tabular, 5000 rows"
        ↓
ChromaDB cosine similarity search
        ↓
Top-K similar experiments retrieved
        ↓
Context injected into:
  • ModelSelectionAgent → better model choice
  • DataCleaningAgent → known preprocessing patterns
  • ConversationalRAGAgent → user questions answered
```

## Components

### `app/memory/embeddings.py` — EmbeddingModel
Local embedding generation using `sentence-transformers`.
- Model: `all-MiniLM-L6-v2` (default) — 80MB, CPU-friendly, excellent quality
- Produces 384-dimensional dense vectors
- Fully offline — no API calls

### `app/memory/chroma_store.py` — ChromaStore
ChromaDB persistence layer:
- Persists to `chroma_persist_dir` (configurable, default: `./chroma_db`)
- Collection: `autonomds_experiments`
- Supports: `add`, `query`, `get_all`, `count`

### `app/memory/experiment_memory.py` — ExperimentMemory
High-level interface combining ChromaStore + EmbeddingModel:
- `store_experiment(id, document, metadata)` → embed + store
- `find_similar(query, n_results)` → semantic search
- `rag_context(query)` → formatted context string for LLM injection

### `app/rag/retrieval.py` — RAGRetriever
Production-grade retrieval with 3 specialised modes:
- `retrieve_for_model_selection(description, task_type)` → recommends models from past success
- `retrieve_for_preprocessing(description)` → recommends cleaning strategies
- `retrieve_for_chat(user_question)` → powers the conversational assistant

## Experiment Document Schema

Each stored document is a natural language summary:

```
"Dataset: titanic.csv (891 rows, 12 cols).
Task: binary_classification.
Target: survived.
Best model: RandomForestClassifier.
Metrics: {'accuracy': 0.83, 'f1': 0.81, 'roc_auc': 0.88}.
Features: 9 selected.
Insights: Class imbalance detected (ratio 2.1x). Missing values in Age (19.9%)."
```

This text is embedded as a vector for similarity search.

## Metadata Schema

Each document also stores structured metadata:

```json
{
  "experiment_id": "exp-abc123",
  "filename":      "titanic.csv",
  "task_type":     "binary_classification",
  "target_column": "survived",
  "n_rows":        891,
  "best_model":    "RandomForestClassifier",
  "best_metrics":  "{\"accuracy\": 0.83}",
  "timestamp":     "2024-01-15T10:30:00Z"
}
```

## Retrieval Quality

The system uses cosine similarity. Quality depends on:
1. **Quality of the document** — more descriptive experiment summaries → better retrieval
2. **Database size** — more experiments stored → more relevant retrievals
3. **Query specificity** — precise queries → better matches

With cold start (0 experiments), all RAG methods return empty context and agents use their defaults.

## Extending RAG

To add new retrieval modes:
1. Add a method to `RAGRetriever` in `app/rag/retrieval.py`
2. Call it in the relevant agent's `execute()` method
3. Inject the returned context string into the LLM prompt

Example:
```python
# In ModelSelectionAgent.execute()
rag = get_retriever()
context = rag.retrieve_for_model_selection(dataset_description, task_type)
llm_prompt = f"Context from past experiments:\n{context}\n\nSelect models for: {task_type}"
```
