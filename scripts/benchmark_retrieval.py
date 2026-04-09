#!/usr/bin/env python3
"""
Benchmark script for task-aware retrieval.

Measures:
1. Reranking latency
2. Task relevance scoring accuracy
3. Recency impact on ranking
"""

import time
from datetime import datetime, timedelta
from uuid import uuid4
import statistics

from src.core.models import Memory, MemoryType
from src.core.retrieval import (
    HybridRetriever,
    RecencyScorer,
    TaskRelevanceScorer,
    RetrievalConfig,
)


def make_memory(
    content: str,
    created_at: datetime | None = None,
    importance: float = 0.5,
    tags: list[str] | None = None,
) -> Memory:
    """Helper to create test memories."""
    return Memory(
        id=uuid4(),
        content=content,
        memory_type=MemoryType.EPISODIC,
        agent_id="benchmark-agent",
        user_id="benchmark-user",
        importance_score=importance,
        tags=tags or [],
        created_at=created_at or datetime.utcnow(),
    )


# Sample memories for benchmarking
SAMPLE_MEMORIES = [
    ("Customer reported order #12345 never arrived, very frustrated", 0.9, ['shipping', 'complaint']),
    ("Refunded $49.99 to customer's Visa ending in 4242", 0.85, ['billing', 'refund']),
    ("Customer thanked agent for quick resolution", 0.3, ['positive']),
    ("Password reset completed successfully", 0.4, ['technical']),
    ("Customer asking about premium subscription features", 0.6, ['billing', 'upgrade']),
    ("Package delayed due to weather, new ETA: Friday", 0.7, ['shipping', 'update']),
    ("Customer mentioned competitor pricing is 20% lower", 0.85, ['retention', 'cancellation']),
    ("Explained return policy: 30 days with receipt", 0.5, ['policy']),
    ("Customer escalated to supervisor, demanded compensation", 0.95, ['complaint', 'escalation']),
    ("Address updated to 123 Main St, Apt 4B", 0.6, ['shipping', 'update']),
]

TASK_CONTEXTS = [
    "Handle shipping delay complaint",
    "Process refund request for subscription",
    "Technical support for login issues",
    "Retention call for customer considering cancellation",
    "Billing inquiry about recent charge",
]


def benchmark_task_relevance_scoring():
    """Benchmark task relevance scoring speed."""
    print("\n=== Task Relevance Scoring ===")
    
    scorer = TaskRelevanceScorer()
    
    # Warmup
    for content, _, tags in SAMPLE_MEMORIES[:3]:
        for task in TASK_CONTEXTS[:2]:
            scorer.score(content, task, tags)
    
    # Benchmark
    iterations = 1000
    latencies = []
    
    for i in range(iterations):
        content, _, tags = SAMPLE_MEMORIES[i % len(SAMPLE_MEMORIES)]
        task = TASK_CONTEXTS[i % len(TASK_CONTEXTS)]
        
        start = time.perf_counter()
        scorer.score(content, task, tags)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    print(f"Iterations: {iterations}")
    print(f"Mean latency: {statistics.mean(latencies):.4f} ms")
    print(f"P50 latency: {statistics.median(latencies):.4f} ms")
    print(f"P99 latency: {sorted(latencies)[int(0.99 * len(latencies))]:.4f} ms")
    print(f"Max latency: {max(latencies):.4f} ms")


def benchmark_recency_scoring():
    """Benchmark recency scoring speed."""
    print("\n=== Recency Scoring ===")
    
    scorer = RecencyScorer()
    now = datetime.utcnow()
    
    # Generate test timestamps
    timestamps = [now - timedelta(days=d) for d in range(0, 100)]
    
    # Benchmark
    iterations = 10000
    latencies = []
    
    for i in range(iterations):
        created_at = timestamps[i % len(timestamps)]
        
        start = time.perf_counter()
        scorer.score(created_at, now)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    print(f"Iterations: {iterations}")
    print(f"Mean latency: {statistics.mean(latencies):.6f} ms")
    print(f"P50 latency: {statistics.median(latencies):.6f} ms")
    print(f"P99 latency: {sorted(latencies)[int(0.99 * len(latencies))]:.6f} ms")


def benchmark_hybrid_reranking():
    """Benchmark full hybrid reranking with various result set sizes."""
    print("\n=== Hybrid Reranking ===")
    
    retriever = HybridRetriever()
    now = datetime.utcnow()
    
    # Test different result set sizes
    for result_count in [5, 10, 20, 50, 100]:
        print(f"\nResult set size: {result_count}")
        
        # Generate test results
        results = []
        for i in range(result_count):
            content, importance, tags = SAMPLE_MEMORIES[i % len(SAMPLE_MEMORIES)]
            results.append({
                'memory': make_memory(
                    content,
                    created_at=now - timedelta(days=i % 30),
                    importance=importance,
                    tags=tags,
                ),
                'similarity_score': 0.5 + (i % 5) * 0.1,
            })
        
        # Warmup
        for _ in range(3):
            retriever.rerank(results.copy(), task_context=TASK_CONTEXTS[0], now=now)
        
        # Benchmark
        iterations = 100
        latencies = []
        
        for i in range(iterations):
            task = TASK_CONTEXTS[i % len(TASK_CONTEXTS)]
            
            start = time.perf_counter()
            retriever.rerank(results.copy(), task_context=task, now=now)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        print(f"  Mean latency: {statistics.mean(latencies):.3f} ms")
        print(f"  P50 latency: {statistics.median(latencies):.3f} ms")
        print(f"  P99 latency: {sorted(latencies)[int(0.99 * len(latencies))]:.3f} ms")


def benchmark_category_extraction():
    """Benchmark category extraction speed."""
    print("\n=== Category Extraction ===")
    
    scorer = TaskRelevanceScorer()
    
    test_texts = [
        "I need a refund for my subscription payment",
        "My package hasn't arrived and I want to track it",
        "I'm frustrated with the service and want to cancel",
        "Help me reset my password, I can't login",
        "What's the price difference between plans?",
    ]
    
    # Warmup
    for text in test_texts:
        scorer.extract_categories(text)
    
    # Benchmark
    iterations = 5000
    latencies = []
    
    for i in range(iterations):
        text = test_texts[i % len(test_texts)]
        
        start = time.perf_counter()
        scorer.extract_categories(text)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    print(f"Iterations: {iterations}")
    print(f"Mean latency: {statistics.mean(latencies):.4f} ms")
    print(f"P50 latency: {statistics.median(latencies):.4f} ms")
    print(f"P99 latency: {sorted(latencies)[int(0.99 * len(latencies))]:.4f} ms")


def main():
    print("=" * 60)
    print("Mnemosyne Retrieval Benchmark")
    print("=" * 60)
    
    benchmark_recency_scoring()
    benchmark_category_extraction()
    benchmark_task_relevance_scoring()
    benchmark_hybrid_reranking()
    
    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
    
    print("\nTarget: Full rerank should be <10ms for 100 results")
    print("This ensures minimal latency impact on memory retrieval.")


if __name__ == "__main__":
    main()
