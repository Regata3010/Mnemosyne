"""Mnemosyne Python SDK for agent integration."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx

from src.core.models import (
    Memory,
    MemoryQuery,
    MemoryType,
    MemoryWrite,
    RetrievedMemory,
    WriteResult,
)


@dataclass
class MnemosyneConfig:
    """SDK configuration."""
    base_url: str = "http://localhost:8000"
    timeout: float = 30.0
    agent_id: str = "default"


class MnemosyneClient:
    """
    Python SDK for Mnemosyne memory service.
    
    Usage:
        client = MnemosyneClient(agent_id="my-agent")
        
        # Write a memory
        result = await client.remember("User prefers email communication")
        
        # Retrieve memories
        memories = await client.recall("How does user prefer to be contacted?")
    """
    
    def __init__(
        self,
        agent_id: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        self.config = MnemosyneConfig(
            base_url=base_url,
            timeout=timeout,
            agent_id=agent_id,
        )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    # === Core Operations ===
    
    async def remember(
        self,
        content: str,
        user_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance_hint: float | None = None,
    ) -> WriteResult:
        """
        Store a memory.
        
        Args:
            content: The memory content to store
            user_id: Optional user identifier for scoping
            session_id: Optional session identifier
            memory_type: Type of memory (episodic, semantic, procedural)
            tags: Optional tags for filtering
            metadata: Optional metadata dict
            importance_hint: Optional hint for importance scoring
        
        Returns:
            WriteResult with memory_id and whether it was stored
        """
        request = MemoryWrite(
            content=content,
            agent_id=self.config.agent_id,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            tags=tags or [],
            metadata=metadata or {},
            importance_hint=importance_hint,
        )
        
        response = await self._client.post(
            "/memories",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
        return WriteResult.model_validate(response.json())
    
    async def recall(
        self,
        query: str,
        user_id: str | None = None,
        task_context: str | None = None,
        memory_types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        min_importance: float | None = None,
        top_k: int = 10,
    ) -> list[RetrievedMemory]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Natural language query
            user_id: Optional user filter
            task_context: Current task for relevance boosting
            memory_types: Filter by memory types
            tags: Filter by tags
            min_importance: Minimum importance score
            top_k: Maximum results to return
        
        Returns:
            List of memories ranked by relevance
        """
        request = MemoryQuery(
            query=query,
            agent_id=self.config.agent_id,
            user_id=user_id,
            task_context=task_context,
            memory_types=memory_types,
            tags=tags,
            min_importance=min_importance,
            top_k=top_k,
        )
        
        response = await self._client.post(
            "/memories/search",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
        return [RetrievedMemory.model_validate(m) for m in response.json()]
    
    async def get(self, memory_id: UUID) -> Memory | None:
        """Get a specific memory by ID."""
        response = await self._client.get(f"/memories/{memory_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Memory.model_validate(response.json())
    
    async def forget(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        response = await self._client.delete(f"/memories/{memory_id}")
        return response.status_code == 204
    
    async def count(self, user_id: str | None = None) -> int:
        """Count memories for this agent."""
        url = f"/memories/count/{self.config.agent_id}"
        if user_id:
            url += f"?user_id={user_id}"
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()["count"]
    
    # === Convenience Methods ===
    
    async def remember_fact(self, content: str, **kwargs) -> WriteResult:
        """Store a semantic (factual) memory."""
        return await self.remember(
            content, 
            memory_type=MemoryType.SEMANTIC, 
            **kwargs
        )
    
    async def remember_interaction(self, content: str, **kwargs) -> WriteResult:
        """Store an episodic (interaction) memory."""
        return await self.remember(
            content, 
            memory_type=MemoryType.EPISODIC, 
            **kwargs
        )
    
    async def remember_procedure(self, content: str, **kwargs) -> WriteResult:
        """Store a procedural (action pattern) memory."""
        return await self.remember(
            content, 
            memory_type=MemoryType.PROCEDURAL, 
            **kwargs
        )
    
    # === Health ===
    
    async def health(self) -> bool:
        """Check if service is healthy."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
