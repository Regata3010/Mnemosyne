"""
Benchmark script for importance scorer latency.

Target: <5ms per scoring operation.
"""

import time
import statistics
from dataclasses import dataclass

from src.core.importance import ImportanceScorer


@dataclass
class BenchmarkResult:
    """Benchmark results."""
    samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def benchmark_importance_scorer(
    num_samples: int = 1000,
    warmup: int = 100,
) -> BenchmarkResult:
    """
    Benchmark importance scorer latency.
    
    Args:
        num_samples: Number of samples to measure
        warmup: Number of warmup iterations (not measured)
    """
    scorer = ImportanceScorer()
    
    # Test cases representing different content types
    test_cases = [
        # Generic (fast path)
        "Hi",
        "Hello there",
        "Thanks",
        "Okay",
        
        # Short important
        "I need a refund",
        "Cancel my subscription",
        
        # Medium length
        "I placed order #12345 for $299.99 last week and it hasn't arrived yet.",
        "I am very frustrated with your service and want to speak to a manager.",
        
        # Long content
        "Hi, my name is John Smith and I've been a customer for 5 years. "
        "I placed an order #98765 on January 15th, 2024 for $1,299.99. "
        "The delivery was scheduled for January 20th but it's now January 25th "
        "and I still haven't received my package. I've called three times already "
        "and each time I was told it would arrive the next day. This is completely "
        "unacceptable and I'm considering canceling my membership and requesting "
        "a full refund. Please escalate this to your supervisor immediately. "
        "You can reach me at john.smith@email.com or 555-123-4567.",
    ]
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        for text in test_cases:
            scorer.score(text)
    
    # Benchmark
    print(f"Benchmarking ({num_samples} samples)...")
    latencies = []
    
    for i in range(num_samples):
        text = test_cases[i % len(test_cases)]
        
        start = time.perf_counter()
        scorer.score(text)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
    
    # Calculate statistics
    latencies.sort()
    
    return BenchmarkResult(
        samples=num_samples,
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        p95_ms=latencies[int(num_samples * 0.95)],
        p99_ms=latencies[int(num_samples * 0.99)],
        min_ms=min(latencies),
        max_ms=max(latencies),
    )


def main():
    """Run benchmark and print results."""
    print("=" * 60)
    print("IMPORTANCE SCORER LATENCY BENCHMARK")
    print("=" * 60)
    print()
    
    result = benchmark_importance_scorer(num_samples=1000, warmup=100)
    
    print(f"Samples:    {result.samples}")
    print(f"Mean:       {result.mean_ms:.3f} ms")
    print(f"Median:     {result.median_ms:.3f} ms")
    print(f"P95:        {result.p95_ms:.3f} ms")
    print(f"P99:        {result.p99_ms:.3f} ms")
    print(f"Min:        {result.min_ms:.3f} ms")
    print(f"Max:        {result.max_ms:.3f} ms")
    print()
    
    # Check against target
    target_ms = 5.0
    if result.p99_ms < target_ms:
        print(f"✅ PASS: P99 latency ({result.p99_ms:.3f}ms) < target ({target_ms}ms)")
    else:
        print(f"❌ FAIL: P99 latency ({result.p99_ms:.3f}ms) >= target ({target_ms}ms)")
        print("   Consider optimizing or quantizing the importance scorer.")


if __name__ == "__main__":
    main()
