"""FastAPI application and routes."""

from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, status

from src.core import get_settings
from src.core.embeddings import EmbeddingService
from src.core.models import (
    Memory,
    MemoryQuery,
    MemoryWrite,
    RetrievedMemory,
    WriteResult,
)
from src.core.service import MemoryService
from src.storage import PostgresStore, RedisCache


# Global service instance
_service: MemoryService | None = None


def get_service() -> MemoryService:
    """Get the memory service instance."""
    if _service is None:
        raise RuntimeError("Service not initialized")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize and cleanup."""
    global _service
    
    settings = get_settings()
    
    # Initialize storage
    postgres = PostgresStore(settings.database_url)
    redis = RedisCache(settings.redis_url)
    embeddings = EmbeddingService(settings.embedding_model)
    
    # Connect
    await postgres.connect()
    await redis.connect()
    
    # Initialize schema
    await postgres.init_schema()
    
    # Create service
    _service = MemoryService(postgres, redis, embeddings)
    
    print(f"🧠 Mnemosyne started on {settings.host}:{settings.port}")
    
    yield
    
    # Cleanup
    await postgres.disconnect()
    await redis.disconnect()
    print("🧠 Mnemosyne shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Mnemosyne",
    description="Production Agent Memory with Lifecycle Management",
    version="0.1.0",
    lifespan=lifespan,
)


# === Health ===

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mnemosyne"}


# === Memory Write ===

@app.post("/memories", response_model=WriteResult, status_code=status.HTTP_201_CREATED)
async def write_memory(request: MemoryWrite) -> WriteResult:
    """
    Write a memory to the store.
    
    The memory will be scored for importance and may be filtered
    if below the threshold.
    """
    service = get_service()
    return await service.write(request)


# === Memory Retrieve ===

@app.post("/memories/search", response_model=list[RetrievedMemory])
async def search_memories(query: MemoryQuery) -> list[RetrievedMemory]:
    """
    Search for memories matching a query.
    
    Supports vector similarity search with optional task-aware reranking.
    """
    service = get_service()
    return await service.retrieve(query)


# === Memory Get/Delete ===

@app.get("/memories/{memory_id}", response_model=Memory)
async def get_memory(memory_id: UUID) -> Memory:
    """Get a single memory by ID."""
    service = get_service()
    memory = await service.get(memory_id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )
    return memory


@app.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: UUID):
    """Delete a memory."""
    service = get_service()
    deleted = await service.delete(memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )


# === Stats ===

@app.get("/memories/count/{agent_id}")
async def count_memories(agent_id: str, user_id: str | None = None):
    """Count memories for an agent/user."""
    service = get_service()
    count = await service.count(agent_id, user_id)
    return {"agent_id": agent_id, "user_id": user_id, "count": count}
