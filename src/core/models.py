"""Core data models for Mnemosyne."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Type of memory being stored."""
    EPISODIC = "episodic"      # Specific interactions/events
    SEMANTIC = "semantic"      # Facts, consolidated knowledge
    PROCEDURAL = "procedural"  # Successful action patterns


class Memory(BaseModel):
    """A single memory unit."""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Core content
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    
    # Context
    agent_id: str
    user_id: str | None = None
    session_id: str | None = None
    
    # Importance & lifecycle
    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    access_count: int = 0
    last_accessed: datetime | None = None
    
    # Metadata
    entities: list[str] = Field(default_factory=list)  # Extracted entities
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Versioning for concurrent writes
    version: int = 1


class MemoryWrite(BaseModel):
    """Request to write a memory."""
    
    content: str
    agent_id: str
    user_id: str | None = None
    session_id: str | None = None
    memory_type: MemoryType = MemoryType.EPISODIC
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Optional: client can provide importance (will be validated/overridden)
    importance_hint: float | None = None


class MemoryQuery(BaseModel):
    """Query for retrieving memories."""
    
    query: str
    agent_id: str
    user_id: str | None = None
    
    # Task context for relevance scoring
    task_context: str | None = None
    
    # Filters
    memory_types: list[MemoryType] | None = None
    tags: list[str] | None = None
    min_importance: float | None = None
    
    # Pagination
    top_k: int = 10
    
    # Time filters
    after: datetime | None = None
    before: datetime | None = None


class RetrievedMemory(BaseModel):
    """A memory with retrieval metadata."""
    
    memory: Memory
    similarity_score: float = 0.0
    relevance_score: float = 0.0  # Combined score after reranking
    
    
class WriteResult(BaseModel):
    """Result of a memory write operation."""
    
    memory_id: UUID
    stored: bool  # False if filtered by importance
    importance_score: float
    filtered_reason: str | None = None  # Why it wasn't stored
