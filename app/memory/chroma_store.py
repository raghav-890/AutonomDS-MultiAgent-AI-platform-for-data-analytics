"""
ChromaDB Vector Store
======================
Persistent ChromaDB client for experiment memory storage and retrieval.
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("chroma_store")


class ChromaStore:
    """Wrapper around ChromaDB for experiment memory."""

    def __init__(self, collection_name: str | None = None) -> None:
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.chroma_collection_name
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.settings.chroma_persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(
        self,
        doc_id: str,
        document: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Add a document to the collection."""
        kwargs: dict[str, Any] = {
            "ids": [doc_id],
            "documents": [document],
            "metadatas": [metadata or {}],
        }
        if embedding:
            kwargs["embeddings"] = [embedding]
        self.collection.upsert(**kwargs)
        logger.debug("chroma_upsert", doc_id=doc_id)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query by text similarity. Returns list of result dicts."""
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, max(1, self.collection.count())),
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return items

    def count(self) -> int:
        return self.collection.count()

    def delete(self, doc_id: str) -> None:
        self.collection.delete(ids=[doc_id])

    def get_all(self) -> list[dict[str, Any]]:
        """Retrieve all documents."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get()
        items = []
        for i in range(len(results["ids"])):
            items.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return items
