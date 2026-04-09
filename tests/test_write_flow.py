"""
Integration test for write flow with importance scoring.
"""

import pytest

from src.core.importance import ImportanceScorer
from src.core.entities import EntityExtractor
from src.core.models import MemoryWrite, MemoryType


class TestWriteFlowIntegration:
    """Test the full write flow with importance scoring and entity extraction."""
    
    @pytest.fixture
    def scorer(self):
        return ImportanceScorer()
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    def test_high_value_message_stored(self, scorer, extractor):
        """High-value message should pass importance filter."""
        content = (
            "Customer John Smith (order #12345) is threatening to cancel "
            "his $500/month subscription due to repeated billing errors. "
            "This is his third complaint this month."
        )
        
        # Score importance
        score, signals = scorer.score(content)
        
        # Extract entities
        entities = extractor.extract(content)
        
        # Should pass threshold (0.3)
        assert score >= 0.3, f"Score {score} should be >= 0.3"
        
        # Should have rich signals
        assert signals.sentiment_intensity > 0.2
        assert signals.actionability > 0.2
        
        # Should have entities
        assert entities.count > 0
        assert len(entities.order_ids) >= 1
        assert len(entities.money) >= 1
    
    def test_low_value_message_filtered(self, scorer, extractor):
        """Low-value message should be filtered out."""
        content = "Hi"
        
        score, signals = scorer.score(content)
        
        # Should be below threshold
        assert score < 0.3, f"Score {score} should be < 0.3 for generic greeting"
    
    def test_medium_value_message(self, scorer, extractor):
        """Medium-value message - has some signals but not urgent."""
        content = "I ordered something last Tuesday and I'm wondering about the shipping status."
        
        score, signals = scorer.score(content)
        
        # Should have some specificity (day mentioned) but be borderline
        # Note: "ordered" is an action keyword
        assert score >= 0.15  # Has some signals
        assert score <= 0.7    # Not urgent/high priority
    
    def test_complaint_with_specifics(self, scorer, extractor):
        """Complaint with specific details should score high."""
        content = (
            "I am extremely frustrated! I ordered item SKU123 on 1/15/2024 "
            "and was charged $199.99 but never received it. I've called "
            "5 times and emailed support@company.com with no response. "
            "I demand a refund immediately or I will dispute the charge "
            "with my credit card company."
        )
        
        score, signals = scorer.score(content)
        entities = extractor.extract(content)
        
        # High score due to multiple signals
        assert score > 0.6
        
        # Strong sentiment
        assert signals.sentiment_intensity > 0.4
        
        # High actionability
        assert signals.actionability > 0.4
        
        # Rich entities
        assert len(entities.money) >= 1
        assert len(entities.emails) >= 1
        assert len(entities.dates) >= 1
    
    def test_positive_feedback_high_value(self, scorer, extractor):
        """Positive feedback should also score reasonably high."""
        content = (
            "I just wanted to say that Sarah in customer service was amazing! "
            "She helped resolve my issue with order #54321 quickly and "
            "professionally. I've been a customer for 10 years and this is "
            "exactly why I keep coming back. Thank you!"
        )
        
        score, signals = scorer.score(content)
        
        # Should score reasonably high due to sentiment and specifics
        assert score > 0.4
        
        # Positive sentiment
        assert signals.raw_sentiment['compound'] > 0.3


class TestEntityMemoryIntegration:
    """Test that extracted entities are properly structured for storage."""
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    def test_entities_for_metadata(self, extractor):
        """Entities should be properly formatted for memory metadata."""
        content = (
            "Customer: Jane Doe (jane@example.com)\n"
            "Order: #99999\n"
            "Amount: $1,250.00\n"
            "Date: 2024-03-15\n"
            "Issue: Product arrived damaged"
        )
        
        entities = extractor.extract(content)
        
        # Should convert to flat list for memory.entities field
        entity_list = entities.to_list()
        assert isinstance(entity_list, list)
        assert len(entity_list) > 0
        
        # Should convert to dict for metadata
        entity_dict = entities.to_dict()
        assert 'money' in entity_dict
        assert 'emails' in entity_dict
        assert 'order_ids' in entity_dict
        
        # Check specific extractions
        assert any('1,250' in m for m in entity_dict['money'])
        assert 'jane@example.com' in entity_dict['emails']
        assert '#99999' in entity_dict['order_ids']
