"""Memory service - core business logic."""

from datetime import datetime
from uuid import UUID

import numpy as np

from src.core import get_settings
from src.core.embeddings import EmbeddingService
from src.core.entities import get_entity_extractor
from src.core.importance import get_importance_scorer
from src.core.models import (
    Memory,
    MemoryQuery,
    MemoryType,
    MemoryWrite,
    RetrievedMemory,
    WriteResult,
)
from src.core.retrieval import get_hybrid_retriever, RetrievalConfig
from src.storage import PostgresStore, RedisCache


class MemoryService:
    """
    Core memory service orchestrating write, retrieve, and lifecycle operations.
    
    This is the main entry point for all memory operations.
    """
    
    def __init__(
        self,
        postgres: PostgresStore,
        redis: RedisCache,
        embeddings: EmbeddingService,
    ):
        self.postgres = postgres
        self.redis = redis
        self.embeddings = embeddings
        self.settings = get_settings()
        
        # Initialize importance scorer and entity extractor
        self.importance_scorer = get_importance_scorer()
        self.entity_extractor = get_entity_extractor()
        
        # Initialize hybrid retriever for task-aware reranking
        self.hybrid_retriever = get_hybrid_retriever()
    
    async def write(self, request: MemoryWrite) -> WriteResult:
        """
        Write a memory to the store.
        
        Steps:
        1. Score importance
        2. Filter if below threshold
        3. Extract entities
        4. Generate embedding
        5. Store in Postgres
        6. Cache in Redis
        """
        # Step 1: Score importance
        importance_score, signals = self.importance_scorer.score(request.content)
        
        # Override with hint if provided and higher
        if request.importance_hint is not None:
            importance_score = max(importance_score, request.importance_hint)
        
        # Step 2: Filter low-importance memories
        if importance_score < self.settings.importance_threshold:
            return WriteResult(
                memory_id=UUID(int=0),  # Null UUID
                stored=False,
                importance_score=importance_score,
                filtered_reason=f"Importance {importance_score:.2f} below threshold {self.settings.importance_threshold}",
            )
        
        # Step 3: Extract entities
        entities = self.entity_extractor.extract(request.content)
        entity_list = entities.to_list()
        
        # Step 4: Create memory object
        memory = Memory(
            content=request.content,
            memory_type=request.memory_type,
            agent_id=request.agent_id,
            user_id=request.user_id,
            session_id=request.session_id,
            importance_score=importance_score,
            entities=entity_list,
            tags=request.tags,
            metadata={
                **request.metadata,
                'importance_signals': {
                    'sentiment': signals.sentiment_intensity,
                    'entities': signals.entity_density,
                    'actionability': signals.actionability,
                    'specificity': signals.specificity,
                },
                'extracted_entities': entities.to_dict(),
            },
        )
        
        # Step 5: Generate embedding
        embedding = self.embeddings.embed(memory.content)
        
        # Step 6: Store in Postgres
        await self.postgres.write(memory, embedding)
        
        # Step 7: Cache in Redis
        await self.redis.cache_memory(memory)
        
        return WriteResult(
            memory_id=memory.id,
            stored=True,
            importance_score=importance_score,
        )
    
    async def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]:
        """
        Retrieve memories matching a query.
        
        Steps:
        1. Generate query embedding
        2. Vector search in Postgres
        3. Apply task-aware reranking (placeholder - Week 3)
        4. Update access counts
        5. Return ranked results
        """
        # Step 1: Generate query embedding
        query_embedding = self.embeddings.embed(query.query)
        
        # Step 2: Vector search
        results = await self.postgres.search(query_embedding, query)
        
        # Step 3: Task-aware reranking (placeholder for Week 3)
        if query.task_context:
            results = await self._rerank_for_task(results, query.task_context)
        
        # Step 4: Update access counts
        for result in results:
            await self.postgres.increment_access(result.memory.id)
        
        return results
    
    async def get(self, memory_id: UUID) -> Memory | None:
        """Get a single memory by ID."""
        # Try cache first
        memory = await self.redis.get_cached_memory(memory_id)
        if memory:
            return memory
        
        # Fall back to Postgres
        memory = await self.postgres.get(memory_id)
        if memory:
            await self.redis.cache_memory(memory)
        return memory
    
    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        await self.redis.invalidate_memory(memory_id)
        return await self.postgres.delete(memory_id)
    
    async def count(self, agent_id: str, user_id: str | None = None) -> int:
        """Count memories for an agent/user."""
        return await self.postgres.count(agent_id, user_id)
    
    async def _rerank_for_task(
        self,
        results: list[RetrievedMemory],
        task_context: str,
    ) -> list[RetrievedMemory]:
        """
        Rerank results based on task context using hybrid scoring.
        
        Combines:
        - Vector similarity (from initial search)
        - Task relevance (keyword/category matching)
        - Recency decay (time-based weighting)
        - Importance score (from ingestion)
        """
        if not results:
            return results
        
        # Convert to dict format for hybrid retriever
        retriever_input = [
            {
                'memory': result.memory,
                'similarity_score': result.relevance_score,
            }
            for result in results
        ]
        
        # Rerank with hybrid scoring
        reranked = self.hybrid_retriever.rerank(
            retriever_input,
            task_context=task_context,
        )
        
        # Convert back to RetrievedMemory
        return [
            RetrievedMemory(
                memory=item['memory'],
                relevance_score=item['relevance_score'],
            )
            for item in reranked
        ]
