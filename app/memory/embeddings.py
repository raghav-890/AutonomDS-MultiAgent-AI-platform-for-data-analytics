"""
Local Embedding Model
======================
sentence-transformers based embedding model.
Runs fully locally — no API calls required.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("embeddings")


class EmbeddingModel:
    """
    Thin wrapper around sentence-transformers for local text embeddings.
    Model is lazily loaded on first use and cached.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = settings.embedding_device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns list of floats."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts efficiently."""
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts."""
        v1 = np.array(self.embed(text1))
        v2 = np.array(self.embed(text2))
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
