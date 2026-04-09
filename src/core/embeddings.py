"""Embedding service for memory vectorization."""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from src.core import get_settings


class EmbeddingService:
    """Generate embeddings for memory content."""
    
    _instance: Optional["EmbeddingService"] = None
    
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().embedding_model
        self._model: SentenceTransformer | None = None
    
    @classmethod
    def get_instance(cls, model_name: str | None = None) -> "EmbeddingService":
        """Get singleton instance for efficiency."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(
            texts, 
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))
    
    def find_similar(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar texts to query."""
        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)
        scores = np.dot(candidate_embs, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx]), candidates[idx]) for idx in top_indices]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


def get_embedder(model_name: str | None = None) -> EmbeddingService:
    """Get embedding service singleton."""
    return EmbeddingService.get_instance(model_name)
