"""
Mnemosyne adapter for benchmark framework.

Wraps Mnemosyne's MemoryService for benchmarking.
"""

from typing import Any

from src.core.embeddings import EmbeddingService
from src.core.models import MemoryQuery, MemoryWrite
from src.core.service import MemoryService
from src.storage import PostgresStore, RedisCache


class MnemosyneAdapter:
    """Adapter to use Mnemosyne with benchmark framework."""
    
    def __init__(self, service: MemoryService):
        self.service = service
    
    async def write(
        self,
        content: str,
        agent_id: str,
        user_id: str,
        metadata: dict,
    ) -> bool:
        """Write memory. Returns True if stored, False if filtered."""
        request = MemoryWrite(
            content=content,
            agent_id=agent_id,
            user_id=user_id,
            metadata=metadata,
        )
        result = await self.service.write(request)
        return result.stored
    
    async def retrieve(
        self,
        query: str,
        agent_id: str,
        user_id: str,
        task_context: str,
        top_k: int,
    ) -> list[dict]:
        """Retrieve relevant memories."""
        request = MemoryQuery(
            query=query,
            agent_id=agent_id,
            user_id=user_id,
            task_context=task_context,
            top_k=top_k,
        )
        results = await self.service.retrieve(request)
        
        return [
            {
                "content": r.memory.content,
                "user_id": r.memory.user_id,
                "metadata": r.memory.metadata,
                "relevance_score": r.relevance_score,
            }
            for r in results
        ]
    
    async def count(self, agent_id: str, user_id: str | None = None) -> int:
        """Count stored memories."""
        return await self.service.count(agent_id, user_id)
    
    async def clear(self, agent_id: str) -> None:
        """Clear all memories for agent."""
        # Get all memories for this agent and delete them
        query = MemoryQuery(
            query="",  # Empty query to get all
            agent_id=agent_id,
            top_k=10000,  # Get all
        )
        # Note: In production, would use a dedicated delete_all method
        pass  # Benchmark will use fresh database


async def create_mnemosyne_adapter() -> MnemosyneAdapter:
    """Create Mnemosyne adapter with real storage."""
    from src.core import get_settings
    
    settings = get_settings()
    
    postgres = PostgresStore(settings.database_url)
    redis = RedisCache(settings.redis_url)
    embeddings = EmbeddingService(settings.embedding_model)
    
    await postgres.connect()
    await redis.connect()
    await postgres.init_schema()
    
    service = MemoryService(postgres, redis, embeddings)
    
    return MnemosyneAdapter(service)
