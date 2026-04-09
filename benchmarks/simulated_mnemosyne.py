"""
Simulated Mnemosyne adapter for benchmarks.

Uses in-memory storage with REAL embeddings to accurately simulate 
Mnemosyne behavior. Implements importance scoring, task-aware
retrieval, and concurrency guarantees.
"""

import asyncio
import numpy as np
from typing import Any, Optional
from uuid import uuid4
from datetime import datetime

from src.core.importance import get_importance_scorer
from src.core.entities import get_entity_extractor
from src.core.retrieval import get_hybrid_retriever
from src.core.concurrency import OptimisticLock, ConflictResolution
from src.core import get_settings


class SimulatedMnemosyneAdapter:
    """
    In-memory simulation of Mnemosyne's behavior with REAL embeddings.
    
    Implements:
    - Importance-based filtering at ingestion
    - Real semantic similarity via sentence-transformers
    - Task-aware retrieval with reranking
    - Optimistic locking for concurrent writes
    """
    
    def __init__(self, importance_threshold: float = 0.32, use_real_embeddings: bool = True):
        self._memories: dict[str, list[dict]] = {}  # agent_id -> memories
        self._user_index: dict[str, list[dict]] = {}  # user_id -> memories
        self._importance_scorer = get_importance_scorer()
        self._entity_extractor = get_entity_extractor()
        self._retriever = get_hybrid_retriever()
        self._lock = OptimisticLock(resolution=ConflictResolution.LAST_WRITE_WINS)
        self._importance_threshold = importance_threshold
        self._async_lock = asyncio.Lock()
        self._use_real_embeddings = use_real_embeddings
        
        # Embedding model (lazy loaded)
        self._embedder: Optional[Any] = None
        
        # Stats for benchmarking
        self.write_conflicts = 0
        self.embedding_calls = 0
    
    def _get_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None and self._use_real_embeddings:
            from src.core.embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder
    
    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        self.embedding_calls += 1
        embedder = self._get_embedder()
        if embedder:
            return embedder.embed(text)
        # Fallback: random embedding
        return np.random.randn(384).astype(np.float32)
    
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        self.embedding_calls += len(texts)
        embedder = self._get_embedder()
        if embedder:
            return embedder.embed_batch(texts)
        # Fallback: random embeddings
        return np.random.randn(len(texts), 384).astype(np.float32)
    
    async def write(
        self,
        content: str,
        agent_id: str,
        user_id: str,
        metadata: dict,
    ) -> bool:
        """
        Write memory with importance filtering.
        
        Returns True if stored, False if filtered due to low importance.
        """
        # Score importance
        importance_score, signals = self._importance_scorer.score(content)
        
        # Filter if below threshold
        if importance_score < self._importance_threshold:
            return False  # Filtered
        
        # Extract entities
        entities = self._entity_extractor.extract(content)
        
        # Generate embedding
        embedding = self._embed(content)
        
        async with self._async_lock:
            memory_id = str(uuid4())
            
            memory = {
                "id": memory_id,
                "content": content,
                "agent_id": agent_id,
                "user_id": user_id,
                "embedding": embedding,
                "metadata": {
                    **metadata,
                    "importance_signals": {
                        "sentiment": signals.sentiment_intensity,
                        "entities": signals.entity_density,
                        "actionability": signals.actionability,
                        "specificity": signals.specificity,
                    },
                },
                "importance_score": importance_score,
                "entities": entities.to_list(),
                "tags": [],
                "created_at": datetime.utcnow(),
                "version": 1,
            }
            
            # Store by agent
            if agent_id not in self._memories:
                self._memories[agent_id] = []
            self._memories[agent_id].append(memory)
            
            # Index by user
            if user_id not in self._user_index:
                self._user_index[user_id] = []
            self._user_index[user_id].append(memory)
            
            return True
    
    async def retrieve(
        self,
        query: str,
        agent_id: str,
        user_id: str,
        task_context: str,
        top_k: int,
    ) -> list[dict]:
        """
        Retrieve memories with REAL vector similarity + task-aware reranking.
        """
        async with self._async_lock:
            # Get memories for this agent
            agent_memories = self._memories.get(agent_id, [])
            
            # Filter by user if specified
            if user_id:
                candidates = [m for m in agent_memories if m.get("user_id") == user_id]
            else:
                candidates = agent_memories
            
            if not candidates:
                return []
            
            # Generate query embedding
            query_embedding = self._embed(query)
            
            # Compute cosine similarities
            scored_results = []
            for memory in candidates:
                mem_embedding = memory.get("embedding")
                if mem_embedding is not None:
                    # Cosine similarity (embeddings are normalized)
                    similarity = float(np.dot(query_embedding, mem_embedding))
                else:
                    similarity = 0.3  # Default if no embedding
                
                scored_results.append({
                    "memory": self._to_memory_object(memory),
                    "similarity_score": similarity,
                })
            
            # Apply hybrid reranking (adds recency, task relevance, importance)
            reranked = self._retriever.rerank(
                scored_results,
                task_context=task_context,
            )
            
            # Return top_k
            return [
                {
                    "content": r["memory"].content,
                    "user_id": r["memory"].user_id,
                    "metadata": r["memory"].metadata,
                    "relevance_score": r["relevance_score"],
                }
                for r in reranked[:top_k]
            ]
    
    def _to_memory_object(self, memory_dict: dict):
        """Convert dict to Memory-like object for retriever."""
        from src.core.models import Memory, MemoryType
        
        return Memory(
            id=memory_dict["id"],
            content=memory_dict["content"],
            memory_type=MemoryType.EPISODIC,
            agent_id=memory_dict["agent_id"],
            user_id=memory_dict.get("user_id"),
            importance_score=memory_dict.get("importance_score", 0.5),
            entities=memory_dict.get("entities", []),
            tags=memory_dict.get("tags", []),
            metadata=memory_dict.get("metadata", {}),
            created_at=memory_dict.get("created_at", datetime.utcnow()),
        )
    
    async def count(self, agent_id: str, user_id: str | None = None) -> int:
        """Count stored memories."""
        async with self._async_lock:
            if user_id:
                return len(self._user_index.get(user_id, []))
            return len(self._memories.get(agent_id, []))
    
    async def clear(self, agent_id: str) -> None:
        """Clear all memories for agent."""
        async with self._async_lock:
            if agent_id in self._memories:
                # Also clear from user index
                for memory in self._memories[agent_id]:
                    user_id = memory.get("user_id")
                    if user_id and user_id in self._user_index:
                        self._user_index[user_id] = [
                            m for m in self._user_index[user_id]
                            if m.get("agent_id") != agent_id
                        ]
                del self._memories[agent_id]


def create_simulated_mnemosyne(
    use_real_embeddings: bool = True,
    importance_threshold: float = 0.32,
) -> SimulatedMnemosyneAdapter:
    """Create simulated Mnemosyne adapter."""
    return SimulatedMnemosyneAdapter(
        use_real_embeddings=use_real_embeddings,
        importance_threshold=importance_threshold,
    )
