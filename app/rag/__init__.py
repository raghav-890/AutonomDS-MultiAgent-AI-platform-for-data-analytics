"""
AutonomDS RAG Module
=====================
Advanced Retrieval-Augmented Generation pipeline for experiment memory.

Architecture:
    Query (user question / dataset description)
        ↓
    EmbeddingModel (local sentence-transformers)
        ↓
    ChromaDB vector search
        ↓
    Similar experiment retrieval
        ↓
    Context-augmented LLM reasoning
        ↓
    Improved agent decisions / user answers

This is NOT a simple PDF chatbot RAG. It is:
- Experiment-memory retrieval for better model selection
- Retrieval-augmented preprocessing decisions
- Conversational querying over stored experiment history
"""
