"""Tests for task-aware retrieval system."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.core.models import Memory, MemoryType
from src.core.retrieval import (
    HybridRetriever,
    RecencyScorer,
    RetrievalConfig,
    TaskRelevanceScorer,
)


# --- RecencyScorer Tests ---

class TestRecencyScorer:
    """Tests for recency-based scoring."""
    
    def test_recent_memory_high_score(self):
        """Very recent memories should score high."""
        scorer = RecencyScorer(half_life_days=7.0, max_boost=0.3)
        now = datetime.utcnow()
        
        # Memory from 1 hour ago
        recent = now - timedelta(hours=1)
        score = scorer.score(recent, now)
        
        assert score > 0.29  # Very close to max_boost
    
    def test_week_old_memory_half_score(self):
        """Memory at half-life should score ~half max_boost."""
        scorer = RecencyScorer(half_life_days=7.0, max_boost=0.3)
        now = datetime.utcnow()
        
        # Memory from 7 days ago
        week_old = now - timedelta(days=7)
        score = scorer.score(week_old, now)
        
        assert 0.14 < score < 0.16  # ~0.15 (half of 0.3)
    
    def test_old_memory_low_score(self):
        """Old memories should score low."""
        scorer = RecencyScorer(half_life_days=7.0, max_boost=0.3)
        now = datetime.utcnow()
        
        # Memory from 30 days ago
        old = now - timedelta(days=30)
        score = scorer.score(old, now)
        
        assert score < 0.05  # Very low
    
    def test_configurable_half_life(self):
        """Different half-life values should change decay rate."""
        now = datetime.utcnow()
        two_days_ago = now - timedelta(days=2)
        
        # Fast decay (1 day half-life)
        fast = RecencyScorer(half_life_days=1.0, max_boost=1.0)
        fast_score = fast.score(two_days_ago, now)
        
        # Slow decay (7 day half-life)
        slow = RecencyScorer(half_life_days=7.0, max_boost=1.0)
        slow_score = slow.score(two_days_ago, now)
        
        assert fast_score < slow_score


# --- TaskRelevanceScorer Tests ---

class TestTaskRelevanceScorer:
    """Tests for task-based relevance scoring."""
    
    def test_billing_task_matches_billing_memory(self):
        """Billing-related memories should match billing tasks."""
        scorer = TaskRelevanceScorer()
        
        task = "Customer asking about refund for subscription"
        memory = "User requested refund for $49.99 annual payment"
        
        score = scorer.score(memory, task)
        assert score > 0.7  # Strong match
    
    def test_shipping_task_matches_shipping_memory(self):
        """Shipping-related memories should match shipping tasks."""
        scorer = TaskRelevanceScorer()
        
        task = "Check status of delayed package delivery"
        memory = "Order #12345 shipped via FedEx, tracking shows transit delay"
        
        score = scorer.score(memory, task)
        assert score > 0.7
    
    def test_no_category_overlap_low_score(self):
        """Memories with no category overlap should score lower."""
        scorer = TaskRelevanceScorer()
        
        task = "Help with password reset"  # Technical
        memory = "Customer's shipping address is 123 Main St"  # Shipping
        
        score = scorer.score(memory, task)
        assert score < 0.5
    
    def test_complaint_detection(self):
        """Complaint-related content should be categorized."""
        scorer = TaskRelevanceScorer()
        
        task = "Handle escalated customer complaint"
        memory = "Customer was very frustrated and demanded to speak with manager"
        
        score = scorer.score(memory, task)
        assert score > 0.7
    
    def test_extract_categories(self):
        """Should correctly extract task categories."""
        scorer = TaskRelevanceScorer()
        
        billing_text = "I need a refund for my subscription payment"
        categories = scorer.extract_categories(billing_text)
        
        assert 'billing' in categories
    
    def test_tags_boost_relevance(self):
        """Memory tags should contribute to relevance."""
        scorer = TaskRelevanceScorer()
        
        task = "Process cancellation request"
        memory = "Customer called about their account"
        tags = ['cancellation', 'urgent']
        
        score_with_tags = scorer.score(memory, task, tags)
        score_without_tags = scorer.score(memory, task)
        
        assert score_with_tags >= score_without_tags


# --- HybridRetriever Tests ---

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
        agent_id="test-agent",
        user_id="test-user",
        importance_score=importance,
        tags=tags or [],
        created_at=created_at or datetime.utcnow(),
    )


class TestHybridRetriever:
    """Tests for combined hybrid retrieval scoring."""
    
    def test_rerank_orders_by_combined_score(self):
        """Results should be ordered by combined relevance score."""
        retriever = HybridRetriever()
        now = datetime.utcnow()
        
        results = [
            {
                'memory': make_memory("old generic content", created_at=now - timedelta(days=30)),
                'similarity_score': 0.8,
            },
            {
                'memory': make_memory("recent billing refund request", created_at=now - timedelta(hours=1)),
                'similarity_score': 0.6,
            },
        ]
        
        reranked = retriever.rerank(results, task_context="Process refund", now=now)
        
        # Recent + task-relevant should beat old + high-similarity
        assert reranked[0]['memory'].content == "recent billing refund request"
    
    def test_recency_boosts_recent_memories(self):
        """Recent memories should get recency boost."""
        retriever = HybridRetriever()
        now = datetime.utcnow()
        
        # Two memories with same content but different ages
        results = [
            {
                'memory': make_memory("same content", created_at=now - timedelta(days=14)),
                'similarity_score': 0.7,
            },
            {
                'memory': make_memory("same content", created_at=now - timedelta(hours=1)),
                'similarity_score': 0.7,
            },
        ]
        
        reranked = retriever.rerank(results, now=now)
        
        # Recent one should score higher
        assert reranked[0]['memory'].created_at > reranked[1]['memory'].created_at
    
    def test_importance_contributes_to_score(self):
        """Higher importance memories should score better."""
        retriever = HybridRetriever()
        now = datetime.utcnow()
        
        results = [
            {
                'memory': make_memory("low importance", importance=0.2),
                'similarity_score': 0.7,
            },
            {
                'memory': make_memory("high importance", importance=0.9),
                'similarity_score': 0.7,
            },
        ]
        
        reranked = retriever.rerank(results, now=now)
        
        high_importance = next(r for r in reranked if "high importance" in r['memory'].content)
        low_importance = next(r for r in reranked if "low importance" in r['memory'].content)
        
        assert high_importance['relevance_score'] > low_importance['relevance_score']
    
    def test_scoring_breakdown_included(self):
        """Reranked results should include scoring breakdown."""
        retriever = HybridRetriever()
        
        results = [
            {
                'memory': make_memory("test content"),
                'similarity_score': 0.8,
            },
        ]
        
        reranked = retriever.rerank(results, task_context="help customer")
        
        assert '_scoring' in reranked[0]
        scoring = reranked[0]['_scoring']
        assert 'similarity' in scoring
        assert 'task_relevance' in scoring
        assert 'recency' in scoring
        assert 'importance' in scoring
    
    def test_no_task_context_neutral_relevance(self):
        """Without task context, task relevance should be neutral."""
        retriever = HybridRetriever()
        
        results = [
            {
                'memory': make_memory("billing content"),
                'similarity_score': 0.8,
            },
        ]
        
        # Without task context
        reranked = retriever.rerank(results)
        
        assert reranked[0]['_scoring']['task_relevance'] == 0.5
    
    def test_empty_results_handled(self):
        """Empty results should return empty list."""
        retriever = HybridRetriever()
        
        reranked = retriever.rerank([])
        assert reranked == []
    
    def test_custom_config(self):
        """Custom config should change scoring weights."""
        # Heavy weight on similarity
        config = RetrievalConfig(
            similarity_weight=0.8,
            task_relevance_weight=0.1,
            recency_weight=0.05,
            importance_weight=0.05,
        )
        retriever = HybridRetriever(config)
        now = datetime.utcnow()
        
        # High similarity, old memory
        high_sim = {
            'memory': make_memory("old content", created_at=now - timedelta(days=30)),
            'similarity_score': 0.95,
        }
        # Low similarity, recent memory
        low_sim = {
            'memory': make_memory("new content", created_at=now - timedelta(hours=1)),
            'similarity_score': 0.3,
        }
        
        reranked = retriever.rerank([low_sim, high_sim], now=now)
        
        # With heavy similarity weight, high-sim should win
        assert reranked[0]['memory'].content == "old content"


class TestRetrieverIntegration:
    """Integration tests for retrieval scenarios."""
    
    def test_customer_service_scenario(self):
        """Test realistic customer service memory retrieval."""
        retriever = HybridRetriever()
        now = datetime.utcnow()
        
        results = [
            # Old greeting
            {
                'memory': make_memory(
                    "Customer said hello",
                    created_at=now - timedelta(days=10),
                    importance=0.2,
                ),
                'similarity_score': 0.4,
            },
            # Recent shipping issue
            {
                'memory': make_memory(
                    "Customer reported package delayed, tracking #TRK123",
                    created_at=now - timedelta(hours=2),
                    importance=0.8,
                    tags=['shipping', 'complaint'],
                ),
                'similarity_score': 0.7,
            },
            # Older billing issue (resolved)
            {
                'memory': make_memory(
                    "Refunded $29.99 to customer's credit card",
                    created_at=now - timedelta(days=5),
                    importance=0.7,
                ),
                'similarity_score': 0.5,
            },
        ]
        
        # Query about shipping
        reranked = retriever.rerank(
            results,
            task_context="Help with package delivery status",
            now=now,
        )
        
        # Shipping memory should be first
        assert "delayed" in reranked[0]['memory'].content
        assert reranked[0]['relevance_score'] > 0.6
    
    def test_returning_customer_scenario(self):
        """Test memory retrieval for returning customer."""
        retriever = HybridRetriever()
        now = datetime.utcnow()
        
        results = [
            # First interaction - subscription signup (old, high similarity due to "subscription")
            {
                'memory': make_memory(
                    "Customer subscribed to Premium plan, $99/year",
                    created_at=now - timedelta(days=60),
                    importance=0.85,
                    tags=['billing', 'subscription'],
                ),
                'similarity_score': 0.5,  # Similar but not as recent/relevant
            },
            # Second interaction - feature question
            {
                'memory': make_memory(
                    "Explained how to export data in CSV format",
                    created_at=now - timedelta(days=30),
                    importance=0.5,
                    tags=['technical', 'support'],
                ),
                'similarity_score': 0.4,
            },
            # Third interaction - cancellation consideration (recent, high importance)
            {
                'memory': make_memory(
                    "Customer considering cancellation due to competitor offer",
                    created_at=now - timedelta(days=7),
                    importance=0.9,
                    tags=['cancellation', 'retention'],
                ),
                'similarity_score': 0.55,  # Slightly higher + more recent + task match
            },
        ]
        
        # Current task: retention call
        reranked = retriever.rerank(
            results,
            task_context="Customer called about cancelling subscription",
            now=now,
        )
        
        # Cancellation memory should be most relevant (recency + task match + importance)
        assert "cancellation" in reranked[0]['memory'].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
