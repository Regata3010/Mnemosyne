"""Tests for entity extractor."""

import pytest

from src.core.entities import EntityExtractor, get_entity_extractor


class TestEntityExtractor:
    """Test entity extraction."""
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    # === Money Extraction ===
    
    def test_extract_money_simple(self, extractor):
        """Extract simple money amounts."""
        result = extractor.extract("The price is $50")
        assert "$50" in result.money
    
    def test_extract_money_with_cents(self, extractor):
        """Extract money with cents."""
        result = extractor.extract("I paid $149.99 for this")
        assert "$149.99" in result.money
    
    def test_extract_money_with_commas(self, extractor):
        """Extract large money amounts."""
        result = extractor.extract("The total was $1,500.00")
        assert "$1,500.00" in result.money
    
    # === Date Extraction ===
    
    def test_extract_date_slash(self, extractor):
        """Extract slash-formatted dates."""
        result = extractor.extract("It was delivered on 1/15/2024")
        assert any("1/15/2024" in d for d in result.dates)
    
    def test_extract_date_iso(self, extractor):
        """Extract ISO-formatted dates."""
        result = extractor.extract("Created on 2024-01-15")
        assert any("2024-01-15" in d for d in result.dates)
    
    # === Order ID Extraction ===
    
    def test_extract_order_id_hash(self, extractor):
        """Extract order IDs with hash."""
        result = extractor.extract("My order #12345 is missing")
        assert "#12345" in result.order_ids
    
    def test_extract_order_id_word(self, extractor):
        """Extract order IDs with 'order' prefix."""
        result = extractor.extract("Order 98765 hasn't shipped")
        assert "#98765" in result.order_ids
    
    # === Email Extraction ===
    
    def test_extract_email(self, extractor):
        """Extract email addresses."""
        result = extractor.extract("Contact me at john.doe@example.com")
        assert "john.doe@example.com" in result.emails
    
    # === Phone Extraction ===
    
    def test_extract_phone_dashes(self, extractor):
        """Extract phone with dashes."""
        result = extractor.extract("Call me at 555-123-4567")
        assert any("555-123-4567" in p for p in result.phones)
    
    def test_extract_phone_parens(self, extractor):
        """Extract phone with parentheses."""
        result = extractor.extract("My number is (555) 123-4567")
        assert len(result.phones) >= 1
    
    # === Combined Extraction ===
    
    def test_extract_multiple_entities(self, extractor):
        """Extract multiple entity types from one text."""
        text = (
            "Hi, I'm John Smith. My order #12345 for $299.99 placed on 1/15/2024 "
            "hasn't arrived. Please contact me at john@email.com or 555-123-4567."
        )
        result = extractor.extract(text)
        
        assert len(result.order_ids) >= 1
        assert len(result.money) >= 1
        assert len(result.emails) >= 1
        assert len(result.phones) >= 1
    
    def test_entity_count(self, extractor):
        """Test total entity count."""
        text = "Order #123 for $50 on 1/1/2024 from john@email.com"
        result = extractor.extract(text)
        
        assert result.count >= 4
    
    def test_to_list(self, extractor):
        """Test flattening to list."""
        text = "Order #123 for $50"
        result = extractor.extract(text)
        
        flat_list = result.to_list()
        assert isinstance(flat_list, list)
        assert len(flat_list) >= 2
    
    def test_to_dict(self, extractor):
        """Test converting to dict."""
        text = "Order #123 for $50"
        result = extractor.extract(text)
        
        d = result.to_dict()
        assert 'money' in d
        assert 'order_ids' in d
        assert isinstance(d['money'], list)
    
    # === Edge Cases ===
    
    def test_empty_string(self, extractor):
        """Empty string should return empty entities."""
        result = extractor.extract("")
        assert result.count == 0
    
    def test_no_entities(self, extractor):
        """Text without entities should return empty."""
        result = extractor.extract("Hello, how are you today?")
        # May have some spaCy entities, but no structured ones
        assert len(result.money) == 0
        assert len(result.order_ids) == 0
        assert len(result.emails) == 0
    
    def test_deduplication(self, extractor):
        """Duplicate entities should be removed."""
        text = "Order #123 and order #123 again"
        result = extractor.extract(text)
        
        # Should only have one #123
        assert result.order_ids.count("#123") == 1


class TestSingletonInstance:
    """Test singleton pattern."""
    
    def test_same_instance(self):
        """Should return same instance."""
        extractor1 = get_entity_extractor()
        extractor2 = get_entity_extractor()
        assert extractor1 is extractor2
