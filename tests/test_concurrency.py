"""Tests for concurrency handling."""

import asyncio
from datetime import datetime
from uuid import uuid4

import pytest

from src.core.concurrency import (
    VersionVector,
    OptimisticLock,
    ConflictResolution,
    ConcurrentWriteGuard,
)


# --- VersionVector Tests ---

class TestVersionVector:
    """Tests for version vector clock."""
    
    def test_increment(self):
        """Incrementing should increase agent's clock."""
        v = VersionVector()
        v2 = v.increment("agent-1")
        
        assert v2.clocks["agent-1"] == 1
        assert "agent-1" not in v.clocks  # Original unchanged
    
    def test_multiple_increments(self):
        """Multiple increments should accumulate."""
        v = VersionVector()
        v = v.increment("agent-1")
        v = v.increment("agent-1")
        v = v.increment("agent-1")
        
        assert v.clocks["agent-1"] == 3
    
    def test_multiple_agents(self):
        """Different agents should have independent clocks."""
        v = VersionVector()
        v = v.increment("agent-1")
        v = v.increment("agent-2")
        v = v.increment("agent-1")
        
        assert v.clocks["agent-1"] == 2
        assert v.clocks["agent-2"] == 1
    
    def test_merge(self):
        """Merge should take max of each component."""
        v1 = VersionVector(clocks={"agent-1": 3, "agent-2": 1})
        v2 = VersionVector(clocks={"agent-1": 1, "agent-2": 5, "agent-3": 2})
        
        merged = v1.merge(v2)
        
        assert merged.clocks["agent-1"] == 3
        assert merged.clocks["agent-2"] == 5
        assert merged.clocks["agent-3"] == 2
    
    def test_happens_before_true(self):
        """Should detect happens-before relationship."""
        v1 = VersionVector(clocks={"agent-1": 1, "agent-2": 1})
        v2 = VersionVector(clocks={"agent-1": 2, "agent-2": 1})
        
        assert v1.happens_before(v2)
        assert not v2.happens_before(v1)
    
    def test_happens_before_false_equal(self):
        """Equal vectors are not happens-before."""
        v1 = VersionVector(clocks={"agent-1": 2, "agent-2": 3})
        v2 = VersionVector(clocks={"agent-1": 2, "agent-2": 3})
        
        assert not v1.happens_before(v2)
        assert not v2.happens_before(v1)
    
    def test_concurrent(self):
        """Should detect concurrent vectors."""
        v1 = VersionVector(clocks={"agent-1": 2, "agent-2": 1})
        v2 = VersionVector(clocks={"agent-1": 1, "agent-2": 2})
        
        assert v1.concurrent_with(v2)
        assert v2.concurrent_with(v1)
    
    def test_serialization(self):
        """Should round-trip through dict."""
        v = VersionVector(clocks={"agent-1": 5, "agent-2": 3})
        
        serialized = v.to_dict()
        restored = VersionVector.from_dict(serialized)
        
        assert restored.clocks == v.clocks


# --- OptimisticLock Tests ---

class TestOptimisticLock:
    """Tests for optimistic locking."""
    
    @pytest.mark.asyncio
    async def test_successful_write(self):
        """Should succeed on first try with no conflict."""
        lock = OptimisticLock()
        
        data = {"value": 1, "version": 1}
        
        async def read_fn():
            return data.copy(), data["version"]
        
        async def write_fn(d, version):
            if version == data["version"]:
                data["value"] = d["value"] + 1
                data["version"] += 1
                return True
            return False
        
        success, retries = await lock.execute_with_retry(read_fn, write_fn)
        
        assert success
        assert retries == 0
        assert data["value"] == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_conflict(self):
        """Should retry on conflict."""
        lock = OptimisticLock(max_retries=3, initial_backoff_ms=1)
        
        version = [1]
        conflict_count = [0]
        
        async def read_fn():
            return {"value": 1}, version[0]
        
        async def write_fn(d, v):
            # First two attempts fail
            if conflict_count[0] < 2:
                conflict_count[0] += 1
                version[0] += 1  # Simulate external modification
                return False
            return True
        
        success, retries = await lock.execute_with_retry(read_fn, write_fn)
        
        assert success
        assert retries == 2
        assert lock.conflicts_detected == 2
    
    @pytest.mark.asyncio
    async def test_fail_resolution(self):
        """Should fail immediately with FAIL resolution."""
        lock = OptimisticLock(resolution=ConflictResolution.FAIL)
        
        async def read_fn():
            return {"value": 1}, 1
        
        async def write_fn(d, v):
            return False  # Always conflict
        
        success, retries = await lock.execute_with_retry(read_fn, write_fn)
        
        assert not success
        assert retries == 0
        assert lock.conflicts_failed == 1
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Should fail after max retries."""
        lock = OptimisticLock(max_retries=2, initial_backoff_ms=1)
        
        async def read_fn():
            return {"value": 1}, 1
        
        async def write_fn(d, v):
            return False  # Always conflict
        
        success, retries = await lock.execute_with_retry(read_fn, write_fn)
        
        assert not success
        assert retries == 3  # Initial + 2 retries + final attempt
        assert lock.conflicts_failed == 1
    
    def test_stats(self):
        """Should track statistics."""
        lock = OptimisticLock()
        
        stats = lock.get_stats()
        
        assert "total_attempts" in stats
        assert "conflicts_detected" in stats
        assert "conflicts_resolved" in stats
        assert "conflicts_failed" in stats


# --- ConcurrentWriteGuard Tests ---

class TestConcurrentWriteGuard:
    """Tests for write guard."""
    
    @pytest.mark.asyncio
    async def test_acquire_local_lock(self):
        """Should acquire local lock."""
        guard = ConcurrentWriteGuard()
        memory_id = uuid4()
        
        acquired = await guard.acquire(memory_id, timeout_seconds=1.0)
        
        assert acquired
        
        await guard.release(memory_id)
    
    @pytest.mark.asyncio
    async def test_lock_blocks_concurrent_acquire(self):
        """Second acquire should wait."""
        guard = ConcurrentWriteGuard()
        memory_id = uuid4()
        
        # First acquire
        await guard.acquire(memory_id)
        
        # Second acquire should timeout
        acquired = await guard.acquire(memory_id, timeout_seconds=0.1)
        
        assert not acquired
        
        # Release and retry
        await guard.release(memory_id)
        acquired = await guard.acquire(memory_id, timeout_seconds=0.1)
        
        assert acquired
        await guard.release(memory_id)
    
    @pytest.mark.asyncio
    async def test_different_memories_independent(self):
        """Different memories should have independent locks."""
        guard = ConcurrentWriteGuard()
        memory1 = uuid4()
        memory2 = uuid4()
        
        await guard.acquire(memory1)
        
        # Should be able to acquire different memory
        acquired = await guard.acquire(memory2, timeout_seconds=0.1)
        
        assert acquired
        
        await guard.release(memory1)
        await guard.release(memory2)


# --- Integration Tests ---

class TestConcurrencyIntegration:
    """Integration tests for concurrent writes."""
    
    @pytest.mark.asyncio
    async def test_concurrent_writers(self):
        """Multiple concurrent writers should not corrupt data."""
        lock = OptimisticLock(max_retries=5, initial_backoff_ms=1)
        
        # Shared state
        counter = {"value": 0, "version": 0}
        write_count = [0]
        
        async def read_fn():
            return {"value": counter["value"]}, counter["version"]
        
        async def write_fn(data, version):
            await asyncio.sleep(0.001)  # Simulate latency
            if version == counter["version"]:
                counter["value"] += 1
                counter["version"] += 1
                write_count[0] += 1
                return True
            return False
        
        # Run 10 concurrent writers
        async def writer():
            await lock.execute_with_retry(read_fn, write_fn)
        
        tasks = [writer() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # All writes should be accounted for
        assert counter["value"] == write_count[0]
        assert counter["version"] == write_count[0]
    
    @pytest.mark.asyncio
    async def test_version_vector_causality(self):
        """Version vectors should track causality correctly."""
        # Simulate distributed system scenario
        
        # Agent 1 writes
        v1 = VersionVector().increment("agent-1")
        
        # Agent 2 observes v1 and writes
        v2 = v1.increment("agent-2")
        
        # Agent 3 writes independently
        v3 = VersionVector().increment("agent-3")
        
        # v1 happens before v2
        assert v1.happens_before(v2)
        
        # v1 and v3 are concurrent
        assert v1.concurrent_with(v3)
        
        # Merge v2 and v3
        merged = v2.merge(v3)
        
        # Merged should dominate both
        assert v2.happens_before(merged) or v2.clocks == merged.clocks
        assert v3.happens_before(merged) or v3.clocks == merged.clocks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
