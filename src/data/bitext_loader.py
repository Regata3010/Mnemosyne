"""
Bitext Customer Support Dataset Loader.
Loads real customer service conversations for benchmarking.
"""

from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import hashlib


@dataclass
class Conversation:
    """A customer service conversation."""
    conversation_id: str
    customer_id: str
    messages: List[Dict[str, Any]]
    category: str
    intent: str
    timestamp: datetime


@dataclass 
class BitextDataset:
    """Loaded Bitext dataset with conversation structure."""
    conversations: List[Conversation]
    categories: List[str]
    intents: List[str]
    total_messages: int
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def sample(self, n: int, seed: int = 42) -> "BitextDataset":
        """Get random sample of conversations."""
        random.seed(seed)
        sampled = random.sample(self.conversations, min(n, len(self.conversations)))
        return BitextDataset(
            conversations=sampled,
            categories=list(set(c.category for c in sampled)),
            intents=list(set(c.intent for c in sampled)),
            total_messages=sum(len(c.messages) for c in sampled)
        )
    
    def by_category(self, category: str) -> List[Conversation]:
        """Filter conversations by category."""
        return [c for c in self.conversations if c.category == category]
    
    def iter_messages(self) -> Iterator[tuple]:
        """Iterate over all messages with metadata."""
        for conv in self.conversations:
            for msg in conv.messages:
                yield conv, msg


def load_bitext_dataset(
    n_conversations: int = 500,
    seed: int = 42,
    simulate_customers: bool = True
) -> BitextDataset:
    """Load Bitext customer support dataset.
    
    Args:
        n_conversations: Number of conversations to load
        seed: Random seed for sampling
        simulate_customers: If True, assign ~30% as returning customers
        
    Returns:
        BitextDataset with structured conversations
    """
    from datasets import load_dataset
    
    # Load from HuggingFace
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )
    
    random.seed(seed)
    
    # Sample conversations
    indices = random.sample(range(len(ds)), min(n_conversations, len(ds)))
    
    conversations = []
    categories = set()
    intents = set()
    total_messages = 0
    
    # Track customer IDs for returning customer simulation
    customer_pool = [f"cust_{i:04d}" for i in range(int(n_conversations * 0.7))]
    
    base_time = datetime.now() - timedelta(days=30)
    
    for i, idx in enumerate(indices):
        row = ds[idx]
        
        # Generate conversation ID
        conv_id = f"conv_{hashlib.md5(row['instruction'].encode()).hexdigest()[:8]}"
        
        # Assign customer ID (some returning)
        if simulate_customers and random.random() < 0.3:
            # Returning customer
            customer_id = random.choice(customer_pool[:len(customer_pool)//3])
        else:
            customer_id = random.choice(customer_pool)
        
        # Create message structure
        messages = [
            {
                "role": "customer",
                "content": row["instruction"],
                "timestamp": base_time + timedelta(hours=i*2)
            },
            {
                "role": "agent", 
                "content": row["response"],
                "timestamp": base_time + timedelta(hours=i*2, minutes=5)
            }
        ]
        
        conv = Conversation(
            conversation_id=conv_id,
            customer_id=customer_id,
            messages=messages,
            category=row["category"],
            intent=row["intent"],
            timestamp=base_time + timedelta(hours=i*2)
        )
        
        conversations.append(conv)
        categories.add(row["category"])
        intents.add(row["intent"])
        total_messages += len(messages)
    
    return BitextDataset(
        conversations=conversations,
        categories=list(categories),
        intents=list(intents),
        total_messages=total_messages
    )


def get_category_stats(dataset: BitextDataset) -> Dict[str, int]:
    """Get conversation count by category."""
    stats = {}
    for conv in dataset.conversations:
        stats[conv.category] = stats.get(conv.category, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: -x[1]))


def get_returning_customer_stats(dataset: BitextDataset) -> Dict[str, Any]:
    """Analyze returning customer patterns."""
    customer_convs = {}
    for conv in dataset.conversations:
        if conv.customer_id not in customer_convs:
            customer_convs[conv.customer_id] = []
        customer_convs[conv.customer_id].append(conv)
    
    returning = {k: v for k, v in customer_convs.items() if len(v) > 1}
    
    return {
        "total_customers": len(customer_convs),
        "returning_customers": len(returning),
        "returning_rate": len(returning) / len(customer_convs) if customer_convs else 0,
        "max_conversations_per_customer": max(len(v) for v in customer_convs.values()) if customer_convs else 0,
        "avg_conversations_per_returning": sum(len(v) for v in returning.values()) / len(returning) if returning else 0
    }
