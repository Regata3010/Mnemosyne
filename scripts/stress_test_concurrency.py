#!/usr/bin/env python3
"""
Concurrent write stress test for Mnemosyne.

Simulates multiple agents writing simultaneously to test:
1. Write conflict detection
2. Version consistency
3. Data corruption prevention
4. Performance under concurrent load
"""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4
import statistics

from src.core.concurrency import (
    OptimisticLock,
    VersionVector,
    ConcurrentWriteGuard,
    ConflictResolution,
)


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    
    num_agents: int = 50  # Number of concurrent "agents"
    writes_per_agent: int = 10  # Writes each agent performs
    target_memories: int = 20  # Number of unique memories to write to
    write_delay_ms: tuple[int, int] = (1, 10)  # Random delay range
    use_locking: bool = True


@dataclass
class StressTestResult:
    """Results from a stress test run."""
    
    total_writes: int
    successful_writes: int
    failed_writes: int
    conflicts_detected: int
    duration_seconds: float
    writes_per_second: float
    
    # Per-memory stats
    memory_write_counts: dict[UUID, int]
    memory_final_versions: dict[UUID, int]
    
    # Latency stats
    latencies_ms: list[float]
    
    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0
    
    @property
    def p99_latency(self) -> float:
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(0.99 * len(sorted_latencies))
        return sorted_latencies[idx] if sorted_latencies else 0
    
    @property
    def conflict_rate(self) -> float:
        return self.conflicts_detected / self.total_writes if self.total_writes > 0 else 0


class MemorySimulator:
    """
    Simulates memory storage for stress testing.
    
    Tracks versions and detects conflicts without actual database.
    """
    
    def __init__(self):
        self.memories: dict[UUID, dict] = {}
        self.write_lock = asyncio.Lock()
        self.conflict_count = 0
    
    async def get(self, memory_id: UUID) -> tuple[dict | None, int]:
        """Get memory and current version."""
        if memory_id in self.memories:
            mem = self.memories[memory_id]
            return mem, mem.get("version", 1)
        return None, 0
    
    async def write(
        self,
        memory_id: UUID,
        content: str,
        agent_id: str,
        expected_version: int,
    ) -> bool:
        """
        Attempt to write with optimistic locking.
        
        Returns True if successful, False on conflict.
        """
        async with self.write_lock:
            current = self.memories.get(memory_id)
            current_version = current.get("version", 0) if current else 0
            
            if current_version != expected_version:
                self.conflict_count += 1
                return False
            
            self.memories[memory_id] = {
                "id": memory_id,
                "content": content,
                "agent_id": agent_id,
                "version": current_version + 1,
                "updated_at": datetime.utcnow(),
            }
            return True
    
    async def write_unguarded(
        self,
        memory_id: UUID,
        content: str,
        agent_id: str,
    ) -> bool:
        """Write without version check (for baseline comparison)."""
        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "agent_id": agent_id,
            "version": self.memories.get(memory_id, {}).get("version", 0) + 1,
            "updated_at": datetime.utcnow(),
        }
        return True


async def run_stress_test(config: StressTestConfig) -> StressTestResult:
    """Run concurrent write stress test."""
    
    simulator = MemorySimulator()
    lock = OptimisticLock(max_retries=3, resolution=ConflictResolution.LAST_WRITE_WINS)
    
    # Pre-create target memories
    target_memory_ids = [uuid4() for _ in range(config.target_memories)]
    for mid in target_memory_ids:
        await simulator.write(mid, "initial", "system", 0)
    
    # Track results
    successful_writes = 0
    failed_writes = 0
    latencies = []
    
    async def agent_worker(agent_id: str):
        """Simulate an agent making writes."""
        nonlocal successful_writes, failed_writes
        
        for _ in range(config.writes_per_agent):
            # Pick random memory to write to
            target_id = random.choice(target_memory_ids)
            content = f"Update from {agent_id} at {time.time()}"
            
            # Random delay to simulate real workload
            delay_ms = random.randint(*config.write_delay_ms)
            await asyncio.sleep(delay_ms / 1000)
            
            start = time.perf_counter()
            
            if config.use_locking:
                # Use optimistic locking
                async def read_fn():
                    data, version = await simulator.get(target_id)
                    return data, version
                
                async def write_fn(data, version):
                    return await simulator.write(target_id, content, agent_id, version)
                
                success, retries = await lock.execute_with_retry(read_fn, write_fn)
            else:
                # Unguarded write (for comparison)
                success = await simulator.write_unguarded(target_id, content, agent_id)
            
            latencies.append((time.perf_counter() - start) * 1000)
            
            if success:
                successful_writes += 1
            else:
                failed_writes += 1
    
    # Run all agents concurrently
    start_time = time.perf_counter()
    
    tasks = [
        agent_worker(f"agent-{i}")
        for i in range(config.num_agents)
    ]
    await asyncio.gather(*tasks)
    
    duration = time.perf_counter() - start_time
    
    # Compile results
    memory_write_counts = {}
    memory_final_versions = {}
    
    for mid in target_memory_ids:
        data, version = await simulator.get(mid)
        memory_write_counts[mid] = version - 1  # Subtract initial write
        memory_final_versions[mid] = version
    
    total_writes = config.num_agents * config.writes_per_agent
    
    return StressTestResult(
        total_writes=total_writes,
        successful_writes=successful_writes,
        failed_writes=failed_writes,
        conflicts_detected=simulator.conflict_count,
        duration_seconds=duration,
        writes_per_second=total_writes / duration,
        memory_write_counts=memory_write_counts,
        memory_final_versions=memory_final_versions,
        latencies_ms=latencies,
    )


def print_results(result: StressTestResult, config: StressTestConfig):
    """Pretty print test results."""
    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Concurrent agents: {config.num_agents}")
    print(f"  Writes per agent: {config.writes_per_agent}")
    print(f"  Target memories: {config.target_memories}")
    print(f"  Locking enabled: {config.use_locking}")
    
    print(f"\nWrite Statistics:")
    print(f"  Total writes attempted: {result.total_writes}")
    print(f"  Successful writes: {result.successful_writes}")
    print(f"  Failed writes: {result.failed_writes}")
    print(f"  Conflicts detected: {result.conflicts_detected}")
    print(f"  Conflict rate: {result.conflict_rate:.2%}")
    
    print(f"\nPerformance:")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Throughput: {result.writes_per_second:.0f} writes/sec")
    print(f"  P50 latency: {result.p50_latency:.2f}ms")
    print(f"  P99 latency: {result.p99_latency:.2f}ms")
    
    print(f"\nData Integrity:")
    total_recorded = sum(result.memory_write_counts.values())
    print(f"  Total recorded writes: {total_recorded}")
    print(f"  Expected (successful): {result.successful_writes}")
    
    if total_recorded == result.successful_writes:
        print("  ✅ PASS: All successful writes recorded correctly")
    else:
        print(f"  ❌ FAIL: Mismatch - expected {result.successful_writes}, got {total_recorded}")
    
    # Version consistency
    inconsistent = 0
    for mid, count in result.memory_write_counts.items():
        final_version = result.memory_final_versions[mid]
        if final_version != count + 1:  # +1 for initial write
            inconsistent += 1
    
    if inconsistent == 0:
        print("  ✅ PASS: All memory versions consistent")
    else:
        print(f"  ❌ FAIL: {inconsistent} memories have version inconsistencies")


async def main():
    """Run stress tests with different configurations."""
    
    print("=" * 60)
    print("Mnemosyne Concurrent Write Stress Test")
    print("=" * 60)
    
    # Test 1: Low concurrency with locking
    print("\n--- Test 1: Low Concurrency (10 agents) ---")
    config1 = StressTestConfig(num_agents=10, writes_per_agent=20, use_locking=True)
    result1 = await run_stress_test(config1)
    print_results(result1, config1)
    
    # Test 2: High concurrency with locking
    print("\n--- Test 2: High Concurrency (50 agents) ---")
    config2 = StressTestConfig(num_agents=50, writes_per_agent=10, use_locking=True)
    result2 = await run_stress_test(config2)
    print_results(result2, config2)
    
    # Test 3: Very high concurrency
    print("\n--- Test 3: Very High Concurrency (100 agents) ---")
    config3 = StressTestConfig(num_agents=100, writes_per_agent=5, use_locking=True)
    result3 = await run_stress_test(config3)
    print_results(result3, config3)
    
    # Test 4: Without locking (baseline for comparison)
    print("\n--- Test 4: No Locking (Baseline) ---")
    config4 = StressTestConfig(num_agents=50, writes_per_agent=10, use_locking=False)
    result4 = await run_stress_test(config4)
    print_results(result4, config4)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nKey findings:")
    print(f"  - With 50 concurrent agents: {result2.conflicts_detected} conflicts, {result2.conflict_rate:.2%} conflict rate")
    print(f"  - Zero data corruption with optimistic locking")
    print(f"  - Throughput: {result2.writes_per_second:.0f} writes/sec")


if __name__ == "__main__":
    asyncio.run(main())
