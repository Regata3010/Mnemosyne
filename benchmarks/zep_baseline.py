"""
Zep Memory Baseline for Benchmarks.

Uses the real Zep client when configured, otherwise falls back to the
simulated baseline for offline benchmarking.

Zep documentation: https://docs.getzep.com/
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from .zep_real import create_real_zep_adapter, should_use_real_zep
except ImportError:  # pragma: no cover - optional dependency / environment config
    create_real_zep_adapter = None

    def should_use_real_zep() -> bool:
        return False


class ZepMemoryAdapter:
    """
    Simulated Zep memory adapter for benchmarking.
    
    Zep characteristics:
    - Stores ALL conversation messages (no filtering)
    - Vector similarity search with metadata filtering
    - Automatic entity extraction
    - Conversation summarization over time
    - Session-based memory organization
    """

    backend_name = "Zep Memory (simulated)"
    
    def __init__(self, use_real_embeddings: bool = True):
        # Session -> messages mapping
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        
        # User -> sessions mapping  
        self._user_sessions: Dict[str, List[str]] = {}
        
        # All memories (agent_id -> memories)
        self._memories: Dict[str, List[Dict[str, Any]]] = {}
        
        # User index
        self._user_index: Dict[str, List[Dict[str, Any]]] = {}
        
        # Embedding settings
        self._use_real_embeddings = use_real_embeddings
        self._embedder = None
        
        # Stats
        self.embedding_calls = 0
        self._async_lock = asyncio.Lock()
    
    def _get_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None and self._use_real_embeddings:
            try:
                from src.core.embeddings import get_embedder
                self._embedder = get_embedder()
            except ImportError:
                pass
        return self._embedder
    
    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        self.embedding_calls += 1
        embedder = self._get_embedder()
        if embedder:
            return embedder.embed(text)
        return np.random.randn(384).astype(np.float32)
    
    async def write(
        self,
        content: str,
        agent_id: str,
        user_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Write memory to Zep.
        
        Zep stores ALL messages - no filtering.
        Returns True always (Zep doesn't filter).
        """
        embedding = self._embed(content)
        
        async with self._async_lock:
            memory_id = str(uuid4())
            
            # Extract entities (simplified - Zep does this automatically)
            entities = self._extract_entities(content)
            
            memory = {
                "id": memory_id,
                "content": content,
                "agent_id": agent_id,
                "user_id": user_id,
                "embedding": embedding,
                "metadata": metadata,
                "entities": entities,
                "created_at": datetime.utcnow(),
                "token_count": len(content.split()),  # Zep tracks tokens
            }
            
            # Store by agent
            if agent_id not in self._memories:
                self._memories[agent_id] = []
            self._memories[agent_id].append(memory)
            
            # Index by user
            if user_id not in self._user_index:
                self._user_index[user_id] = []
            self._user_index[user_id].append(memory)
            
            return True  # Zep always stores
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Simple entity extraction (Zep does this automatically)."""
        entities = []
        
        # Order IDs
        import re
        order_matches = re.findall(r'#?\d{4,}', text)
        for match in order_matches:
            entities.append({"type": "order_id", "value": match})
        
        # Money
        money_matches = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        for match in money_matches:
            entities.append({"type": "money", "value": match})
        
        # Emails
        email_matches = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
        for match in email_matches:
            entities.append({"type": "email", "value": match})
        
        return entities
    
    async def retrieve(
        self,
        query: str,
        agent_id: str,
        user_id: str,
        task_context: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories from Zep using vector similarity.
        
        Zep uses pure vector similarity (no importance reranking).
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
            
            # Score by cosine similarity (pure vector search)
            scored = []
            for memory in candidates:
                mem_embedding = memory.get("embedding")
                if mem_embedding is not None:
                    similarity = float(np.dot(query_embedding, mem_embedding))
                else:
                    similarity = 0.0
                
                scored.append((memory, similarity))
            
            # Sort by similarity
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k
            return [
                {
                    "content": m["content"],
                    "user_id": m["user_id"],
                    "metadata": m.get("metadata", {}),
                    "relevance_score": score,
                }
                for m, score in scored[:top_k]
            ]
    
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
                for memory in self._memories[agent_id]:
                    user_id = memory.get("user_id")
                    if user_id and user_id in self._user_index:
                        self._user_index[user_id] = [
                            m for m in self._user_index[user_id]
                            if m.get("agent_id") != agent_id
                        ]
                del self._memories[agent_id]
    
    # Zep-specific features
    
    async def get_session_summary(self, session_id: str) -> Optional[str]:
        """
        Get conversation summary for a session.
        
        Zep automatically generates summaries.
        """
        messages = self._sessions.get(session_id, [])
        if not messages:
            return None
        
        # Simple summary (Zep uses LLM)
        content = " ".join(m.get("content", "")[:50] for m in messages[:5])
        return f"Conversation summary: {content[:200]}..."
    
    async def search_memories(
        self,
        query: str,
        agent_id: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search across all memories (Zep's search API).
        """
        return await self.retrieve(
            query=query,
            agent_id=agent_id,
            user_id=None,  # Search all users
            task_context="",
            top_k=limit,
        )


def create_zep_adapter(
    use_real_embeddings: bool = True,
    use_real: bool | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ZepMemoryAdapter:
    """Create Zep memory adapter.

    Uses the real Zep backend when explicitly requested or when the
    environment indicates a configured Zep deployment.
    """
    if use_real is None:
        use_real = should_use_real_zep()

    if use_real:
        if create_real_zep_adapter is None:
            raise ImportError(
                "Real Zep requested but zep-python is not available in this environment."
            )
        return create_real_zep_adapter(
            use_real_embeddings=use_real_embeddings,
            base_url=base_url,
            api_key=api_key,
        )

    return ZepMemoryAdapter(use_real_embeddings=use_real_embeddings)
