"""Tests for core models."""

from uuid import UUID

import pytest

from src.core.models import (
    Memory,
    MemoryQuery,
    MemoryType,
    MemoryWrite,
    RetrievedMemory,
    WriteResult,
)


class TestMemory:
    """Test Memory model."""
    
    def test_create_memory(self):
        """Test creating a memory with defaults."""
        memory = Memory(
            content="User prefers email communication",
            agent_id="test-agent",
        )
        
        assert memory.content == "User prefers email communication"
        assert memory.agent_id == "test-agent"
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.importance_score == 0.5
        assert memory.version == 1
        assert isinstance(memory.id, UUID)
    
    def test_memory_with_all_fields(self):
        """Test memory with all fields specified."""
        memory = Memory(
            content="Customer complained about billing 3 times",
            agent_id="support-agent",
            user_id="user-123",
            session_id="sess-456",
            memory_type=MemoryType.SEMANTIC,
            importance_score=0.9,
            entities=["billing", "complaint"],
            tags=["high-priority", "billing"],
            metadata={"complaint_count": 3},
        )
        
        assert memory.user_id == "user-123"
        assert memory.memory_type == MemoryType.SEMANTIC
        assert memory.importance_score == 0.9
        assert "billing" in memory.entities


class TestMemoryWrite:
    """Test MemoryWrite request model."""
    
    def test_minimal_write_request(self):
        """Test minimal write request."""
        request = MemoryWrite(
            content="Test content",
            agent_id="test-agent",
        )
        
        assert request.content == "Test content"
        assert request.memory_type == MemoryType.EPISODIC
        assert request.importance_hint is None
    
    def test_write_with_importance_hint(self):
        """Test write request with importance hint."""
        request = MemoryWrite(
            content="Critical information",
            agent_id="test-agent",
            importance_hint=0.95,
        )
        
        assert request.importance_hint == 0.95


class TestMemoryQuery:
    """Test MemoryQuery model."""
    
    def test_minimal_query(self):
        """Test minimal query."""
        query = MemoryQuery(
            query="What does the user prefer?",
            agent_id="test-agent",
        )
        
        assert query.query == "What does the user prefer?"
        assert query.top_k == 10
        assert query.task_context is None
    
    def test_query_with_task_context(self):
        """Test query with task context for reranking."""
        query = MemoryQuery(
            query="Customer history",
            agent_id="support-agent",
            task_context="Handling refund request",
            min_importance=0.5,
            top_k=5,
        )
        
        assert query.task_context == "Handling refund request"
        assert query.min_importance == 0.5


class TestWriteResult:
    """Test WriteResult model."""
    
    def test_stored_result(self):
        """Test result when memory was stored."""
        result = WriteResult(
            memory_id=UUID("12345678-1234-5678-1234-567812345678"),
            stored=True,
            importance_score=0.8,
        )
        
        assert result.stored is True
        assert result.filtered_reason is None
    
    def test_filtered_result(self):
        """Test result when memory was filtered."""
        result = WriteResult(
            memory_id=UUID(int=0),
            stored=False,
            importance_score=0.2,
            filtered_reason="Importance below threshold",
        )
        
        assert result.stored is False
        assert "threshold" in result.filtered_reason
