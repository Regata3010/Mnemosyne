"""
LangChain buffer memory baseline for benchmarks.

Uses LangChain's ConversationBufferMemory as baseline comparison.
This represents standard "store everything" approach.
"""

import asyncio
from typing import Any
from uuid import uuid4


class LangChainBufferBaseline:
    """
    Simulates LangChain ConversationBufferMemory behavior.
    
    Key characteristics:
    - Stores ALL messages (no filtering)
    - Simple recency-based retrieval
    - No importance scoring
    - No task-aware reranking
    """
    
    def __init__(self, embedding_service=None):
        self._memories: dict[str, list[dict]] = {}  # agent_id -> memories
        self._user_memories: dict[str, list[dict]] = {}  # user_id -> memories
        self._embedding_service = embedding_service
        self._lock = asyncio.Lock()
    
    async def write(
        self,
        content: str,
        agent_id: str,
        user_id: str,
        metadata: dict,
    ) -> bool:
        """
        Write memory. LangChain stores EVERYTHING (no filtering).
        Always returns True.
        """
        async with self._lock:
            memory_id = str(uuid4())
            
            memory = {
                "id": memory_id,
                "content": content,
                "agent_id": agent_id,
                "user_id": user_id,
                "metadata": metadata,
                "timestamp": asyncio.get_event_loop().time(),
            }
            
            # Store by agent
            if agent_id not in self._memories:
                self._memories[agent_id] = []
            self._memories[agent_id].append(memory)
            
            # Index by user
            if user_id not in self._user_memories:
                self._user_memories[user_id] = []
            self._user_memories[user_id].append(memory)
            
            return True  # LangChain always stores
    
    async def retrieve(
        self,
        query: str,
        agent_id: str,
        user_id: str,
        task_context: str,
        top_k: int,
    ) -> list[dict]:
        """
        Retrieve memories. LangChain uses simple similarity or recency.
        
        Without embeddings: Returns most recent memories for user/agent.
        With embeddings: Would do vector similarity (not implemented here).
        """
        async with self._lock:
            # Get memories for this agent
            agent_memories = self._memories.get(agent_id, [])
            
            # Filter by user if specified
            if user_id:
                user_memories = [m for m in agent_memories if m.get("user_id") == user_id]
            else:
                user_memories = agent_memories
            
            # Sort by timestamp (most recent first) - LangChain buffer behavior
            sorted_memories = sorted(
                user_memories,
                key=lambda m: m.get("timestamp", 0),
                reverse=True,
            )
            
            # Return top_k
            results = sorted_memories[:top_k]
            
            return [
                {
                    "content": m["content"],
                    "user_id": m.get("user_id"),
                    "metadata": m.get("metadata", {}),
                    "relevance_score": 1.0 - (i * 0.1),  # Decay by position
                }
                for i, m in enumerate(results)
            ]
    
    async def count(self, agent_id: str, user_id: str | None = None) -> int:
        """Count stored memories."""
        async with self._lock:
            if user_id:
                return len(self._user_memories.get(user_id, []))
            return len(self._memories.get(agent_id, []))
    
    async def clear(self, agent_id: str) -> None:
        """Clear all memories for agent."""
        async with self._lock:
            if agent_id in self._memories:
                # Also clear from user index
                for memory in self._memories[agent_id]:
                    user_id = memory.get("user_id")
                    if user_id and user_id in self._user_memories:
                        self._user_memories[user_id] = [
                            m for m in self._user_memories[user_id]
                            if m.get("agent_id") != agent_id
                        ]
                del self._memories[agent_id]


def create_langchain_baseline() -> LangChainBufferBaseline:
    """Create LangChain baseline instance."""
    return LangChainBufferBaseline()
