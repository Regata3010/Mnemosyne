"""Redis storage for hot cache and working memory."""

import json
from datetime import timedelta
from typing import Any
from uuid import UUID

import redis.asyncio as redis

from src.core import get_settings
from src.core.models import Memory


class RedisCache:
    """Redis cache for hot memory access."""
    
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or get_settings().redis_url
        self._client: redis.Redis | None = None
        
        # Key prefixes
        self.MEMORY_PREFIX = "mem:"
        self.WORKING_PREFIX = "work:"
        self.LOCK_PREFIX = "lock:"
    
    async def connect(self) -> None:
        """Initialize Redis connection."""
        self._client = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client."""
        if not self._client:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client
    
    # === Hot Memory Cache ===
    
    async def cache_memory(
        self, 
        memory: Memory, 
        ttl: timedelta = timedelta(hours=1),
    ) -> None:
        """Cache a memory for fast retrieval."""
        key = f"{self.MEMORY_PREFIX}{memory.id}"
        await self.client.setex(
            key,
            ttl,
            memory.model_dump_json(),
        )
    
    async def get_cached_memory(self, memory_id: UUID) -> Memory | None:
        """Get a memory from cache."""
        key = f"{self.MEMORY_PREFIX}{memory_id}"
        data = await self.client.get(key)
        if data:
            return Memory.model_validate_json(data)
        return None
    
    async def invalidate_memory(self, memory_id: UUID) -> None:
        """Remove a memory from cache."""
        key = f"{self.MEMORY_PREFIX}{memory_id}"
        await self.client.delete(key)
    
    # === Working Memory (Current Session Context) ===
    
    async def set_working_memory(
        self,
        agent_id: str,
        session_id: str,
        context: dict[str, Any],
        ttl: timedelta = timedelta(hours=2),
    ) -> None:
        """Store working memory for current session."""
        key = f"{self.WORKING_PREFIX}{agent_id}:{session_id}"
        await self.client.setex(key, ttl, json.dumps(context))
    
    async def get_working_memory(
        self,
        agent_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get working memory for current session."""
        key = f"{self.WORKING_PREFIX}{agent_id}:{session_id}"
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def append_to_working_memory(
        self,
        agent_id: str,
        session_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Append to a list in working memory."""
        wm_key = f"{self.WORKING_PREFIX}{agent_id}:{session_id}"
        current = await self.get_working_memory(agent_id, session_id) or {}
        
        if key not in current:
            current[key] = []
        current[key].append(value)
        
        await self.client.setex(
            wm_key,
            timedelta(hours=2),
            json.dumps(current),
        )
    
    async def clear_working_memory(self, agent_id: str, session_id: str) -> None:
        """Clear working memory for a session."""
        key = f"{self.WORKING_PREFIX}{agent_id}:{session_id}"
        await self.client.delete(key)
    
    # === Distributed Locking ===
    
    async def acquire_lock(
        self,
        resource_id: str,
        holder_id: str,
        ttl: timedelta = timedelta(seconds=30),
    ) -> bool:
        """
        Acquire a distributed lock.
        
        Returns True if lock acquired, False if already held.
        """
        key = f"{self.LOCK_PREFIX}{resource_id}"
        # SET NX = only set if not exists
        result = await self.client.set(
            key,
            holder_id,
            ex=ttl,
            nx=True,
        )
        return result is not None
    
    async def release_lock(self, resource_id: str, holder_id: str) -> bool:
        """
        Release a distributed lock.
        
        Only releases if we hold the lock (fencing).
        """
        key = f"{self.LOCK_PREFIX}{resource_id}"
        
        # Lua script for atomic check-and-delete
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        result = await self.client.eval(script, 1, key, holder_id)
        return result == 1
    
    async def extend_lock(
        self,
        resource_id: str,
        holder_id: str,
        ttl: timedelta = timedelta(seconds=30),
    ) -> bool:
        """Extend lock TTL if we hold it."""
        key = f"{self.LOCK_PREFIX}{resource_id}"
        
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        result = await self.client.eval(
            script, 1, key, holder_id, int(ttl.total_seconds())
        )
        return result == 1
