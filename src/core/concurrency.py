"""
Concurrency handling for Mnemosyne.

Provides:
1. Version vectors for conflict detection
2. Optimistic locking with retry logic
3. Conflict resolution strategies
4. Concurrent write stress test harness
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from uuid import UUID

from src.core.models import Memory


class ConflictResolution(str, Enum):
    """Strategy for resolving write conflicts."""
    LAST_WRITE_WINS = "last_write_wins"  # Default: newest timestamp wins
    FIRST_WRITE_WINS = "first_write_wins"  # Original version preserved
    MERGE = "merge"  # Attempt to merge (for compatible changes)
    FAIL = "fail"  # Raise error on conflict


@dataclass
class VersionVector:
    """
    Vector clock for tracking causal ordering of writes.
    
    Each agent maintains a logical clock. When agent A writes,
    it increments its counter. The vector captures "happens-before"
    relationships across distributed writers.
    """
    
    # Map of agent_id -> logical_clock
    clocks: dict[str, int] = field(default_factory=dict)
    
    def increment(self, agent_id: str) -> "VersionVector":
        """Increment clock for agent and return new vector."""
        new_clocks = self.clocks.copy()
        new_clocks[agent_id] = new_clocks.get(agent_id, 0) + 1
        return VersionVector(clocks=new_clocks)
    
    def merge(self, other: "VersionVector") -> "VersionVector":
        """Merge two vectors, taking max of each component."""
        merged = self.clocks.copy()
        for agent_id, clock in other.clocks.items():
            merged[agent_id] = max(merged.get(agent_id, 0), clock)
        return VersionVector(clocks=merged)
    
    def happens_before(self, other: "VersionVector") -> bool:
        """
        Check if this vector causally precedes other.
        
        Returns True if all our clocks <= other's clocks,
        and at least one is strictly less.
        """
        all_leq = True
        any_less = False
        
        all_agents = set(self.clocks.keys()) | set(other.clocks.keys())
        
        for agent_id in all_agents:
            our_clock = self.clocks.get(agent_id, 0)
            their_clock = other.clocks.get(agent_id, 0)
            
            if our_clock > their_clock:
                all_leq = False
            if our_clock < their_clock:
                any_less = True
        
        return all_leq and any_less
    
    def concurrent_with(self, other: "VersionVector") -> bool:
        """Check if vectors are concurrent (neither happens-before the other)."""
        return not self.happens_before(other) and not other.happens_before(self)
    
    def to_dict(self) -> dict[str, int]:
        """Serialize to dict for storage."""
        return self.clocks.copy()
    
    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "VersionVector":
        """Deserialize from dict."""
        return cls(clocks=data.copy())


@dataclass
class WriteConflict:
    """Represents a detected write conflict."""
    
    memory_id: UUID
    current_version: int
    attempted_version: int
    current_content: str
    attempted_content: str
    conflict_time: datetime = field(default_factory=datetime.utcnow)
    resolution: ConflictResolution | None = None
    resolved: bool = False


class OptimisticLock:
    """
    Optimistic locking with configurable retry behavior.
    
    Uses version numbers to detect concurrent modifications.
    Retries with exponential backoff on conflict.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff_ms: int = 10,
        max_backoff_ms: int = 1000,
        resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
    ):
        self.max_retries = max_retries
        self.initial_backoff_ms = initial_backoff_ms
        self.max_backoff_ms = max_backoff_ms
        self.resolution = resolution
        
        # Statistics
        self.total_attempts = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        self.conflicts_failed = 0
    
    async def execute_with_retry(
        self,
        read_fn: Callable[[], Any],
        write_fn: Callable[[Any, int], bool],
        merge_fn: Callable[[Any, Any], Any] | None = None,
    ) -> tuple[bool, int]:
        """
        Execute a read-modify-write operation with retry.
        
        Args:
            read_fn: Async function that returns (data, version)
            write_fn: Async function that takes (data, expected_version) and returns success bool
            merge_fn: Optional function to merge conflicting data
        
        Returns:
            Tuple of (success, retry_count)
        """
        retries = 0
        backoff_ms = self.initial_backoff_ms
        
        while retries <= self.max_retries:
            self.total_attempts += 1
            
            # Read current state
            data, version = await read_fn()
            
            # Attempt write
            success = await write_fn(data, version)
            
            if success:
                return True, retries
            
            # Conflict detected
            self.conflicts_detected += 1
            
            if self.resolution == ConflictResolution.FAIL:
                self.conflicts_failed += 1
                return False, retries
            
            # Retry with backoff
            retries += 1
            if retries <= self.max_retries:
                await asyncio.sleep(backoff_ms / 1000)
                backoff_ms = min(backoff_ms * 2, self.max_backoff_ms)
        
        self.conflicts_failed += 1
        return False, retries
    
    def get_stats(self) -> dict[str, int]:
        """Get conflict statistics."""
        return {
            "total_attempts": self.total_attempts,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "conflicts_failed": self.conflicts_failed,
        }


class ConcurrentWriteGuard:
    """
    Guards concurrent writes to the same memory.
    
    Uses Redis distributed locks for coordination across instances.
    Falls back to in-memory locks for single-instance deployment.
    """
    
    def __init__(self, redis_cache=None):
        self.redis = redis_cache
        self._local_locks: dict[UUID, asyncio.Lock] = {}
    
    def _get_local_lock(self, memory_id: UUID) -> asyncio.Lock:
        """Get or create local lock for memory."""
        if memory_id not in self._local_locks:
            self._local_locks[memory_id] = asyncio.Lock()
        return self._local_locks[memory_id]
    
    async def acquire(self, memory_id: UUID, timeout_seconds: float = 5.0) -> bool:
        """
        Acquire lock for memory.
        
        Tries Redis lock first, falls back to local lock.
        """
        if self.redis:
            # Try distributed lock
            lock_key = f"memory_lock:{memory_id}"
            acquired = await self.redis.acquire_lock(lock_key, timeout_seconds)
            if acquired:
                return True
        
        # Fallback to local lock
        lock = self._get_local_lock(memory_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout_seconds)
            return True
        except asyncio.TimeoutError:
            return False
    
    async def release(self, memory_id: UUID) -> None:
        """Release lock for memory."""
        if self.redis:
            lock_key = f"memory_lock:{memory_id}"
            await self.redis.release_lock(lock_key)
        
        # Also release local lock if held
        if memory_id in self._local_locks:
            lock = self._local_locks[memory_id]
            if lock.locked():
                lock.release()


# Singleton instances
_write_guard: ConcurrentWriteGuard | None = None


def get_write_guard(redis_cache=None) -> ConcurrentWriteGuard:
    """Get singleton write guard."""
    global _write_guard
    if _write_guard is None:
        _write_guard = ConcurrentWriteGuard(redis_cache)
    return _write_guard
