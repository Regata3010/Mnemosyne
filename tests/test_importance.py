"""Tests for importance scorer."""

import pytest

from src.core.importance import ImportanceScorer, get_importance_scorer


class TestImportanceScorer:
    """Test the importance scoring system."""
    
    @pytest.fixture
    def scorer(self):
        return ImportanceScorer()
    
    # === Generic/Low-Value Content ===
    
    def test_generic_greeting_low_score(self, scorer):
        """Generic greetings should score low."""
        score, _ = scorer.score("Hi")
        assert score < 0.3
        
        score, _ = scorer.score("Hello")
        assert score < 0.3
        
        score, _ = scorer.score("How are you")
        assert score < 0.3
    
    def test_generic_acknowledgment_low_score(self, scorer):
        """Generic acknowledgments should score low."""
        score, _ = scorer.score("Okay")
        assert score < 0.3
        
        score, _ = scorer.score("Thanks")
        assert score < 0.3
        
        score, _ = scorer.score("Got it")
        assert score < 0.3
    
    # === High-Value Content ===
    
    def test_complaint_high_score(self, scorer):
        """Complaints should score higher than generic content."""
        score, signals = scorer.score(
            "I am very frustrated with your service. This is the third time "
            "I've had to call about this billing issue."
        )
        assert score > 0.3  # Above typical threshold
        assert signals.sentiment_intensity > 0.3
        assert signals.actionability > 0.3
    
    def test_money_entities_high_score(self, scorer):
        """Content with money should score higher."""
        score, signals = scorer.score(
            "I was charged $150.00 on my credit card but I only ordered $50 worth of items."
        )
        assert score > 0.25  # Above threshold
        assert len(signals.entities_found) >= 2
    
    def test_cancellation_high_score(self, scorer):
        """Cancellation threats should score high."""
        score, signals = scorer.score(
            "If this isn't resolved today, I'm going to cancel my subscription "
            "and request a full refund."
        )
        assert score > 0.3  # Should be above default threshold
        assert 'cancel' in signals.action_keywords_found
        assert 'refund' in signals.action_keywords_found
    
    def test_order_id_has_entity(self, scorer):
        """Content with order IDs should have entity detected."""
        score, signals = scorer.score(
            "My order #12345 was supposed to arrive on January 15th but it's still not here."
        )
        # Should detect the order ID entity even if overall score is low
        assert len(signals.entities_found) >= 1
        assert any('#12345' in e for e in signals.entities_found)
    
    # === Sentiment Detection ===
    
    def test_positive_sentiment_detected(self, scorer):
        """Strong positive sentiment should be detected."""
        score, signals = scorer.score(
            "This is absolutely amazing! Best customer service I've ever experienced!"
        )
        assert signals.sentiment_intensity > 0.5
        assert signals.raw_sentiment['compound'] > 0.5
    
    def test_negative_sentiment_detected(self, scorer):
        """Strong negative sentiment should be detected."""
        score, signals = scorer.score(
            "This is terrible. I'm extremely disappointed and angry."
        )
        assert signals.sentiment_intensity > 0.5
        assert signals.raw_sentiment['compound'] < -0.3
    
    # === Actionability Detection ===
    
    def test_urgent_request_detected(self, scorer):
        """Urgent requests should be detected."""
        score, signals = scorer.score(
            "I need this fixed ASAP. Please prioritize this issue."
        )
        assert signals.actionability > 0.5
        assert 'need' in signals.action_keywords_found
        assert 'asap' in signals.action_keywords_found
    
    def test_deadline_detected(self, scorer):
        """Deadlines should be detected."""
        score, signals = scorer.score(
            "The deadline for this project is next Friday. We must have it done."
        )
        assert signals.actionability > 0.3
        assert 'deadline' in signals.action_keywords_found
        assert 'must' in signals.action_keywords_found
    
    # === Edge Cases ===
    
    def test_empty_string(self, scorer):
        """Empty string should score low."""
        score, _ = scorer.score("")
        assert score < 0.3
    
    def test_whitespace_only(self, scorer):
        """Whitespace should score low."""
        score, _ = scorer.score("   \n\t  ")
        assert score < 0.3
    
    def test_mixed_content(self, scorer):
        """Mixed content should score based on signals."""
        score, signals = scorer.score(
            "Hi there. I placed order #98765 for $299.99 last week "
            "and it hasn't shipped yet. I need it by Friday or I'll have to cancel."
        )
        # Should have multiple positive signals, score above threshold
        assert score > 0.35
        assert len(signals.entities_found) >= 2
        assert len(signals.action_keywords_found) >= 2


class TestSingletonInstance:
    """Test singleton pattern."""
    
    def test_same_instance(self):
        """Should return same instance."""
        scorer1 = get_importance_scorer()
        scorer2 = get_importance_scorer()
        assert scorer1 is scorer2
