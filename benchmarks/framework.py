"""
Benchmark Suite for Mnemosyne

Compares memory systems on:
1. Recall Precision - What fraction of retrieved memories are actually relevant?
2. Storage Efficiency - How many memories stored to achieve same task performance?
3. Consistency Under Load - Zero write conflicts with concurrent agents?
4. Task Performance - Better outcomes for returning customers?

Uses Bitext Customer Support dataset (27K real utterances).
"""

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np


@dataclass
class Conversation:
    """A customer service conversation."""
    conversation_id: str
    messages: list[dict[str, str]]  # role, content
    category: str  # intent category from Bitext
    customer_id: str | None = None
    is_returning: bool = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    
    # Dataset
    num_conversations: int = 500
    returning_customer_ratio: float = 0.3  # 30% are returning customers
    seed: int = 42
    
    # Concurrency test
    concurrent_agents: int = 50
    writes_per_agent: int = 10
    
    # Retrieval test
    top_k: int = 5
    relevance_threshold: float = 0.5


@dataclass
class SystemMetrics:
    """Metrics for a single memory system."""
    
    # Recall precision
    total_retrievals: int = 0
    relevant_retrievals: int = 0
    
    # Storage efficiency
    memories_stored: int = 0
    memories_filtered: int = 0
    
    # Concurrency
    write_conflicts: int = 0
    write_corruptions: int = 0
    
    # Task performance
    returning_customer_tasks: int = 0
    returning_customer_success: int = 0
    
    @property
    def recall_precision(self) -> float:
        if self.total_retrievals == 0:
            return 0.0
        return self.relevant_retrievals / self.total_retrievals
    
    @property
    def storage_efficiency(self) -> float:
        total = self.memories_stored + self.memories_filtered
        if total == 0:
            return 0.0
        return self.memories_filtered / total  # Higher = more efficient filtering
    
    @property
    def task_success_rate(self) -> float:
        if self.returning_customer_tasks == 0:
            return 0.0
        return self.returning_customer_success / self.returning_customer_tasks


class MemorySystem(Protocol):
    """Protocol for memory system implementations."""
    
    async def write(self, content: str, agent_id: str, user_id: str, metadata: dict) -> bool:
        """Write memory. Returns True if stored, False if filtered."""
        ...
    
    async def retrieve(self, query: str, agent_id: str, user_id: str, task_context: str, top_k: int) -> list[dict]:
        """Retrieve relevant memories."""
        ...
    
    async def count(self, agent_id: str, user_id: str | None = None) -> int:
        """Count stored memories."""
        ...
    
    async def clear(self, agent_id: str) -> None:
        """Clear all memories for agent."""
        ...


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    system_name: str
    config: BenchmarkConfig
    metrics: SystemMetrics
    duration_seconds: float
    
    # Detailed breakdowns
    per_category_precision: dict[str, float] = field(default_factory=dict)
    latency_stats: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "metrics": {
                "recall_precision": self.metrics.recall_precision,
                "storage_efficiency": self.metrics.storage_efficiency,
                "write_conflicts": self.metrics.write_conflicts,
                "task_success_rate": self.metrics.task_success_rate,
                "memories_stored": self.metrics.memories_stored,
                "memories_filtered": self.metrics.memories_filtered,
            },
            "duration_seconds": self.duration_seconds,
            "per_category_precision": self.per_category_precision,
            "latency_stats": self.latency_stats,
        }


def load_bitext_dataset(
    num_samples: int = 500,
    seed: int = 42,
    use_real: bool = True,
) -> list[dict]:
    """
    Load Bitext customer support dataset.
    
    Dataset: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
    
    Args:
        num_samples: Number of conversations to load
        seed: Random seed for reproducible sampling
        use_real: If True, load real Bitext data. If False, use synthetic.
    """
    random.seed(seed)

    if not use_real:
        print("Using synthetic data (use_real=False)")
        return generate_synthetic_data(num_samples, seed=seed)
    
    try:
        from datasets import load_dataset
        
        print(f"Loading Bitext dataset ({num_samples} samples)...")
        
        # Load dataset
        dataset = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split="train",
        )
        
        print(f"  Total available: {len(dataset)} conversations")
        print(f"  Categories: {len(set(d['category'] for d in dataset))}")
        
        # Sample conversations
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            samples = [dataset[i] for i in indices]
        else:
            samples = list(dataset)
        
        print(f"  Sampled: {len(samples)} conversations")
        return samples
        
    except ImportError:
        print("Warning: datasets library not installed. Using synthetic data.")
        return generate_synthetic_data(num_samples, seed=seed)
    except Exception as e:
        print(f"Warning: Could not load Bitext dataset: {e}. Using synthetic data.")
        return generate_synthetic_data(num_samples, seed=seed)


def generate_synthetic_data(num_samples: int, seed: int = 42) -> list[dict]:
    """Generate synthetic customer service data for testing."""
    random.seed(seed)
    
    categories = [
        "refund_request", "order_status", "billing_issue",
        "technical_support", "cancellation", "complaint",
        "account_access", "shipping_delay", "product_inquiry",
    ]
    
    templates = {
        "refund_request": [
            "I need a refund for order #{order_id}. I was charged ${amount} but the item was defective.",
            "Please refund my purchase. The product doesn't work as advertised.",
            "I want my money back. This is not what I ordered.",
        ],
        "order_status": [
            "Where is my order #{order_id}? I placed it {days} days ago.",
            "Can you check the status of my delivery? Tracking shows no updates.",
            "My package hasn't arrived yet. When will it be delivered?",
        ],
        "billing_issue": [
            "I was double-charged ${amount} on my credit card.",
            "There's an unauthorized charge of ${amount} on my account.",
            "My subscription was charged incorrectly this month.",
        ],
        "technical_support": [
            "I can't log into my account. The password reset isn't working.",
            "The app keeps crashing when I try to checkout.",
            "I'm getting an error message when I try to update my profile.",
        ],
        "cancellation": [
            "I want to cancel my subscription immediately.",
            "Please cancel my order #{order_id} before it ships.",
            "How do I cancel my premium membership?",
        ],
        "complaint": [
            "This is unacceptable! I've been waiting {days} days for a response.",
            "I'm very frustrated with your customer service.",
            "Your service has been terrible. I want to speak to a manager.",
        ],
        "account_access": [
            "I forgot my password and can't access my account.",
            "My account is locked. Can you help me unlock it?",
            "I need to update the email address on my account.",
        ],
        "shipping_delay": [
            "My order is delayed. The original delivery date was {date}.",
            "Why is my package taking so long? It's been {days} days.",
            "The tracking says my package is stuck in transit.",
        ],
        "product_inquiry": [
            "Do you have this item in a different color?",
            "Is this product compatible with my device?",
            "What's the difference between the basic and premium versions?",
        ],
    }
    
    samples = []
    for i in range(num_samples):
        category = random.choice(categories)
        template = random.choice(templates[category])
        
        # Fill in template variables
        content = template.format(
            order_id=random.randint(10000, 99999),
            amount=random.randint(20, 500),
            days=random.randint(1, 14),
            date=f"{random.randint(1,12)}/{random.randint(1,28)}/2024",
        )
        
        samples.append({
            "instruction": content,
            "category": category,
            "intent": category,
        })
    
    return samples


def create_conversations(
    samples: list[dict],
    returning_ratio: float = 0.3,
    seed: int = 42,
) -> list[Conversation]:
    """Convert samples into conversation objects with customer IDs."""
    
    random.seed(seed)

    conversations = []
    
    # Create pool of customer IDs
    num_unique_customers = int(len(samples) * (1 - returning_ratio))
    customer_ids = [f"customer-{i}" for i in range(num_unique_customers)]
    
    # Track which customers have been seen
    seen_customers = set()
    
    for i, sample in enumerate(samples):
        # Assign customer ID
        if random.random() < returning_ratio and seen_customers:
            # Returning customer
            customer_id = random.choice(list(seen_customers))
            is_returning = True
        else:
            # New customer
            customer_id = customer_ids[i % len(customer_ids)]
            seen_customers.add(customer_id)
            is_returning = False
        
        conversation = Conversation(
            conversation_id=f"conv-{i}",
            messages=[
                {"role": "customer", "content": sample.get("instruction", sample.get("text", ""))},
            ],
            category=sample.get("category", sample.get("intent", "general")),
            customer_id=customer_id,
            is_returning=is_returning,
        )
        conversations.append(conversation)
    
    return conversations


class BenchmarkRunner:
    """Runs benchmarks across different memory systems."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: list[BenchmarkResult] = []
    
    async def run_recall_precision_test(
        self,
        system: MemorySystem,
        conversations: list[Conversation],
    ) -> tuple[SystemMetrics, dict[str, float]]:
        """
        Test recall precision: What fraction of retrieved memories are relevant?
        
        Method:
        1. For each conversation, store the message
        2. Later, query with related task
        3. Check if retrieved memories are actually relevant to the task
        """
        metrics = SystemMetrics()
        per_category = {}
        category_counts = {}
        
        agent_id = "benchmark-agent"
        
        # Phase 1: Store memories from conversations
        for conv in conversations:
            for msg in conv.messages:
                if msg["role"] == "customer":
                    stored = await system.write(
                        content=msg["content"],
                        agent_id=agent_id,
                        user_id=conv.customer_id,
                        metadata={"category": conv.category, "conversation_id": conv.conversation_id},
                    )
                    if stored:
                        metrics.memories_stored += 1
                    else:
                        metrics.memories_filtered += 1
        
        # Phase 2: Query and evaluate relevance
        for conv in conversations:
            # Create a query for this conversation's category
            task_context = f"Help customer with {conv.category.replace('_', ' ')}"
            query_text = conv.messages[0]["content"][:100]  # Use first 100 chars
            
            results = await system.retrieve(
                query=query_text,
                agent_id=agent_id,
                user_id=conv.customer_id,
                task_context=task_context,
                top_k=self.config.top_k,
            )
            
            # Evaluate: Are retrieved memories relevant to this category?
            relevant_count = 0
            for result in results:
                metrics.total_retrievals += 1
                
                # A memory is "relevant" if it's from same category or same customer
                result_category = result.get("metadata", {}).get("category", "")
                result_customer = result.get("user_id", "")
                
                if result_category == conv.category or result_customer == conv.customer_id:
                    relevant_count += 1
                    metrics.relevant_retrievals += 1
            
            # Track per-category precision
            if conv.category not in category_counts:
                category_counts[conv.category] = {"total": 0, "relevant": 0}
            category_counts[conv.category]["total"] += len(results)
            category_counts[conv.category]["relevant"] += relevant_count
        
        # Calculate per-category precision
        for cat, counts in category_counts.items():
            if counts["total"] > 0:
                per_category[cat] = counts["relevant"] / counts["total"]
        
        return metrics, per_category
    
    async def run_concurrency_test(
        self,
        system: MemorySystem,
    ) -> SystemMetrics:
        """
        Test concurrent write consistency.
        
        50 agents writing simultaneously to shared memory space.
        """
        metrics = SystemMetrics()
        agent_id = "benchmark-agent"
        
        async def agent_worker(worker_id: int):
            for i in range(self.config.writes_per_agent):
                content = f"Message from agent {worker_id}, iteration {i}"
                try:
                    await system.write(
                        content=content,
                        agent_id=agent_id,
                        user_id=f"user-{worker_id}",
                        metadata={"worker_id": worker_id, "iteration": i},
                    )
                except Exception as e:
                    if "conflict" in str(e).lower():
                        metrics.write_conflicts += 1
                    else:
                        metrics.write_corruptions += 1
        
        # Run all agents concurrently
        tasks = [agent_worker(i) for i in range(self.config.concurrent_agents)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return metrics
    
    async def run_task_performance_test(
        self,
        system: MemorySystem,
        conversations: list[Conversation],
    ) -> SystemMetrics:
        """
        Test task performance on returning customers.
        
        Does having memory of previous interactions improve outcomes?
        """
        metrics = SystemMetrics()
        agent_id = "benchmark-agent"
        
        # First, store all conversations
        for conv in conversations:
            for msg in conv.messages:
                if msg["role"] == "customer":
                    await system.write(
                        content=msg["content"],
                        agent_id=agent_id,
                        user_id=conv.customer_id,
                        metadata={"category": conv.category},
                    )
        
        # Test returning customers
        returning_convs = [c for c in conversations if c.is_returning]
        
        for conv in returning_convs:
            metrics.returning_customer_tasks += 1
            
            # Query for this customer's history
            results = await system.retrieve(
                query=conv.messages[0]["content"][:100],
                agent_id=agent_id,
                user_id=conv.customer_id,
                task_context=f"Help returning customer with {conv.category}",
                top_k=self.config.top_k,
            )
            
            # Success = found relevant prior context
            if results:
                for result in results:
                    if result.get("user_id") == conv.customer_id:
                        metrics.returning_customer_success += 1
                        break
        
        return metrics
    
    async def benchmark_system(
        self,
        system: MemorySystem,
        system_name: str,
        conversations: list[Conversation],
    ) -> BenchmarkResult:
        """Run full benchmark suite on a memory system."""
        
        print(f"\n{'='*60}")
        print(f"Benchmarking: {system_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Clear previous data
        await system.clear("benchmark-agent")
        
        # Run tests
        print("Running recall precision test...")
        precision_metrics, per_category = await self.run_recall_precision_test(system, conversations)
        
        await system.clear("benchmark-agent")
        
        print("Running concurrency test...")
        concurrency_metrics = await self.run_concurrency_test(system)
        
        await system.clear("benchmark-agent")
        
        print("Running task performance test...")
        task_metrics = await self.run_task_performance_test(system, conversations)
        
        duration = time.time() - start_time
        
        # Combine metrics
        combined_metrics = SystemMetrics(
            total_retrievals=precision_metrics.total_retrievals,
            relevant_retrievals=precision_metrics.relevant_retrievals,
            memories_stored=precision_metrics.memories_stored,
            memories_filtered=precision_metrics.memories_filtered,
            write_conflicts=concurrency_metrics.write_conflicts,
            write_corruptions=concurrency_metrics.write_corruptions,
            returning_customer_tasks=task_metrics.returning_customer_tasks,
            returning_customer_success=task_metrics.returning_customer_success,
        )
        
        result = BenchmarkResult(
            system_name=system_name,
            config=self.config,
            metrics=combined_metrics,
            duration_seconds=duration,
            per_category_precision=per_category,
        )
        
        self.results.append(result)
        
        return result
    
    def print_results(self):
        """Print comparison of all results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS COMPARISON")
        print("=" * 80)
        
        # Header
        print(f"\n{'System':<20} {'Recall':<12} {'Efficiency':<12} {'Conflicts':<12} {'Task Rate':<12}")
        print("-" * 68)
        
        for result in self.results:
            m = result.metrics
            recall = f"{m.recall_precision:.2%}"
            efficiency = f"{m.storage_efficiency:.2%}"
            conflicts = str(m.write_conflicts)
            task_rate = f"{m.task_success_rate:.2%}"
            print(f"{result.system_name:<20} {recall:<12} {efficiency:<12} {conflicts:<12} {task_rate:<12}")
        
        print("\n" + "-" * 68)
        print("Legend:")
        print("  Recall: Fraction of retrieved memories that are relevant")
        print("  Efficiency: Fraction of memories filtered at ingestion")
        print("  Conflicts: Number of write conflicts detected")
        print("  Task Rate: Success rate on returning customer tasks")
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "num_conversations": self.config.num_conversations,
                "concurrent_agents": self.config.concurrent_agents,
                "top_k": self.config.top_k,
            },
            "results": [r.to_dict() for r in self.results],
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
