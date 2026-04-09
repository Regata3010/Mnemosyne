"""PostgreSQL storage with pgvector for memory persistence."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from src.core import get_settings
from src.core.models import Memory, MemoryType, MemoryQuery, RetrievedMemory


class PostgresStore:
    """PostgreSQL + pgvector storage backend."""
    
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or get_settings().database_url
        self._pool: asyncpg.Pool | None = None
        self._embedding_dim = get_settings().embedding_dimension
    
    async def connect(self) -> None:
        """Initialize connection pool."""
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            init=self._init_connection,
        )
    
    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize each connection with pgvector."""
        await register_vector(conn)
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Acquire a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            yield conn
    
    async def init_schema(self) -> None:
        """Create tables and indexes."""
        async with self.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Main memories table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type VARCHAR(20) NOT NULL,
                    agent_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255),
                    session_id VARCHAR(255),
                    importance_score FLOAT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    entities TEXT[] DEFAULT ARRAY[]::TEXT[],
                    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({self._embedding_dim}),
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    version INTEGER NOT NULL DEFAULT 1
                );
            """)
            
            # Indexes for common queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_agent_id 
                ON memories(agent_id);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user_id 
                ON memories(user_id);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance 
                ON memories(importance_score DESC);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created 
                ON memories(created_at DESC);
            """)
            
            # Vector similarity index (IVFFlat for faster search)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding 
                ON memories 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Composite index for agent+user queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_agent_user 
                ON memories(agent_id, user_id);
            """)
    
    async def write(
        self, 
        memory: Memory, 
        embedding: np.ndarray,
    ) -> UUID:
        """
        Write a memory to the database.
        
        Uses optimistic locking - if version conflict, raises error.
        """
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (
                    id, content, memory_type, agent_id, user_id, session_id,
                    importance_score, access_count, last_accessed,
                    entities, tags, metadata, embedding,
                    created_at, updated_at, version
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                )
                """,
                memory.id,
                memory.content,
                memory.memory_type.value,
                memory.agent_id,
                memory.user_id,
                memory.session_id,
                memory.importance_score,
                memory.access_count,
                memory.last_accessed,
                memory.entities,
                memory.tags,
                memory.metadata,
                embedding,
                memory.created_at,
                memory.updated_at,
                memory.version,
            )
            return memory.id
    
    async def update(
        self,
        memory_id: UUID,
        updates: dict,
        expected_version: int,
        new_embedding: np.ndarray | None = None,
    ) -> bool:
        """
        Update a memory with optimistic locking.
        
        Returns True if update succeeded, False if version conflict.
        """
        async with self.acquire() as conn:
            # Build dynamic update query
            set_clauses = ["updated_at = NOW()", "version = version + 1"]
            params = [memory_id, expected_version]
            param_idx = 3
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ${param_idx}")
                params.append(value)
                param_idx += 1
            
            if new_embedding is not None:
                set_clauses.append(f"embedding = ${param_idx}")
                params.append(new_embedding)
            
            result = await conn.execute(
                f"""
                UPDATE memories 
                SET {', '.join(set_clauses)}
                WHERE id = $1 AND version = $2
                """,
                *params,
            )
            
            # Check if row was updated
            return result.split()[-1] == "1"
    
    async def get(self, memory_id: UUID) -> Memory | None:
        """Get a single memory by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memories WHERE id = $1",
                memory_id,
            )
            if row:
                return self._row_to_memory(row)
            return None
    
    async def search(
        self,
        query_embedding: np.ndarray,
        query: MemoryQuery,
    ) -> list[RetrievedMemory]:
        """
        Search memories by vector similarity with filters.
        
        Returns memories sorted by similarity score.
        """
        async with self.acquire() as conn:
            # Build WHERE clauses
            conditions = ["agent_id = $2"]
            params: list = [query_embedding, query.agent_id]
            param_idx = 3
            
            if query.user_id:
                conditions.append(f"user_id = ${param_idx}")
                params.append(query.user_id)
                param_idx += 1
            
            if query.memory_types:
                conditions.append(f"memory_type = ANY(${param_idx})")
                params.append([mt.value for mt in query.memory_types])
                param_idx += 1
            
            if query.tags:
                conditions.append(f"tags && ${param_idx}")
                params.append(query.tags)
                param_idx += 1
            
            if query.min_importance:
                conditions.append(f"importance_score >= ${param_idx}")
                params.append(query.min_importance)
                param_idx += 1
            
            if query.after:
                conditions.append(f"created_at >= ${param_idx}")
                params.append(query.after)
                param_idx += 1
            
            if query.before:
                conditions.append(f"created_at <= ${param_idx}")
                params.append(query.before)
                param_idx += 1
            
            where_clause = " AND ".join(conditions)
            
            # Query with cosine similarity
            rows = await conn.fetch(
                f"""
                SELECT *, 
                       1 - (embedding <=> $1) as similarity_score
                FROM memories
                WHERE {where_clause}
                ORDER BY embedding <=> $1
                LIMIT {query.top_k}
                """,
                *params,
            )
            
            results = []
            for row in rows:
                memory = self._row_to_memory(row)
                results.append(RetrievedMemory(
                    memory=memory,
                    similarity_score=row["similarity_score"],
                    relevance_score=row["similarity_score"],  # Will be reranked later
                ))
            
            return results
    
    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        async with self.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1",
                memory_id,
            )
            return result.split()[-1] == "1"
    
    async def increment_access(self, memory_id: UUID) -> None:
        """Increment access count and update last_accessed."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                UPDATE memories 
                SET access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE id = $1
                """,
                memory_id,
            )
    
    async def count(self, agent_id: str, user_id: str | None = None) -> int:
        """Count memories for an agent/user."""
        async with self.acquire() as conn:
            if user_id:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE agent_id = $1 AND user_id = $2",
                    agent_id, user_id,
                )
            return await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE agent_id = $1",
                agent_id,
            )
    
    def _row_to_memory(self, row: asyncpg.Record) -> Memory:
        """Convert a database row to a Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            importance_score=row["importance_score"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            entities=row["entities"] or [],
            tags=row["tags"] or [],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=row["version"],
        )
