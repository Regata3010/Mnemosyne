"""
Entity Extractor - extracts structured entities from text.

Uses spaCy NER + regex patterns for:
- People, organizations, locations
- Money, dates, times
- Product names, order IDs
- Contact info (emails, phones)
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedEntities:
    """Structured entities extracted from text."""
    
    # Named entities from spaCy
    persons: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    
    # Regex-extracted values
    money: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    order_ids: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    
    def to_list(self) -> list[str]:
        """Flatten all entities to a single list."""
        all_entities = []
        for entities in [
            self.persons, self.organizations, self.locations,
            self.products, self.money, self.dates, self.times,
            self.order_ids, self.emails, self.phones
        ]:
            all_entities.extend(entities)
        return all_entities
    
    def to_dict(self) -> dict[str, list[str]]:
        """Convert to dictionary."""
        return {
            'persons': self.persons,
            'organizations': self.organizations,
            'locations': self.locations,
            'products': self.products,
            'money': self.money,
            'dates': self.dates,
            'times': self.times,
            'order_ids': self.order_ids,
            'emails': self.emails,
            'phones': self.phones,
        }
    
    @property
    def count(self) -> int:
        """Total entity count."""
        return len(self.to_list())


class EntityExtractor:
    """
    Extracts named entities and structured values from text.
    
    Combines spaCy NER with regex patterns for comprehensive extraction.
    """
    
    # Regex patterns for structured data
    PATTERNS = {
        'money': r'\$[\d,]+(?:\.\d{2})?',
        'date_slash': r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        'date_iso': r'\b\d{4}-\d{2}-\d{2}\b',
        'date_written': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b',
        'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        'order_id': r'(?:#|order\s*#?|ticket\s*#?|case\s*#?)\s*(\d+)',
        'product_code': r'\b[A-Z]{2,}\d+[A-Z0-9]*\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    }
    
    # spaCy entity label mapping
    SPACY_LABELS = {
        'PERSON': 'persons',
        'ORG': 'organizations',
        'GPE': 'locations',      # Geo-political entity
        'LOC': 'locations',
        'PRODUCT': 'products',
        'WORK_OF_ART': 'products',
        'MONEY': 'money',
        'DATE': 'dates',
        'TIME': 'times',
    }
    
    def __init__(self):
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }
        self._spacy_nlp = None
    
    @property
    def spacy_nlp(self):
        """Lazy load spaCy model."""
        if self._spacy_nlp is None:
            import spacy
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                self._spacy_nlp = spacy.blank("en")
        return self._spacy_nlp
    
    def extract(self, content: str) -> ExtractedEntities:
        """
        Extract all entities from content.
        
        Args:
            content: Text to extract entities from
        
        Returns:
            ExtractedEntities with all found entities
        """
        result = ExtractedEntities()
        
        # Regex extraction (fast)
        self._extract_regex(content, result)
        
        # spaCy NER (comprehensive)
        self._extract_spacy(content, result)
        
        # Deduplicate
        self._deduplicate(result)
        
        return result
    
    def _extract_regex(self, content: str, result: ExtractedEntities) -> None:
        """Extract entities using regex patterns."""
        
        # Money
        for match in self._compiled_patterns['money'].findall(content):
            result.money.append(match)
        
        # Dates (multiple patterns)
        for pattern_name in ['date_slash', 'date_iso', 'date_written']:
            for match in self._compiled_patterns[pattern_name].findall(content):
                result.dates.append(match)
        
        # Times
        for match in self._compiled_patterns['time'].findall(content):
            result.times.append(match)
        
        # Order IDs
        for match in self._compiled_patterns['order_id'].findall(content):
            result.order_ids.append(f"#{match}")
        
        # Product codes (but filter out common false positives)
        for match in self._compiled_patterns['product_code'].findall(content):
            if not self._is_common_acronym(match):
                result.products.append(match)
        
        # Emails
        for match in self._compiled_patterns['email'].findall(content):
            result.emails.append(match.lower())
        
        # Phones
        for match in self._compiled_patterns['phone'].findall(content):
            result.phones.append(match)
    
    def _extract_spacy(self, content: str, result: ExtractedEntities) -> None:
        """Extract named entities using spaCy."""
        try:
            doc = self.spacy_nlp(content)
            
            for ent in doc.ents:
                if ent.label_ in self.SPACY_LABELS:
                    target_field = self.SPACY_LABELS[ent.label_]
                    target_list = getattr(result, target_field)
                    target_list.append(ent.text)
        except Exception:
            pass  # spaCy failed, continue with regex results
    
    def _deduplicate(self, result: ExtractedEntities) -> None:
        """Remove duplicate entities."""
        for field_name in ['persons', 'organizations', 'locations', 'products',
                          'money', 'dates', 'times', 'order_ids', 'emails', 'phones']:
            field_list = getattr(result, field_name)
            # Preserve order while deduplicating
            seen = set()
            deduped = []
            for item in field_list:
                item_lower = item.lower()
                if item_lower not in seen:
                    seen.add(item_lower)
                    deduped.append(item)
            setattr(result, field_name, deduped)
    
    def _is_common_acronym(self, text: str) -> bool:
        """Check if text is a common acronym (false positive filter)."""
        common = {'AM', 'PM', 'USA', 'UK', 'NYC', 'LA', 'SF', 'TX', 'CA', 'NY',
                 'CEO', 'CTO', 'CFO', 'VP', 'ID', 'URL', 'API', 'FAQ', 'TBD'}
        return text.upper() in common


# Singleton instance
_extractor: EntityExtractor | None = None


def get_entity_extractor() -> EntityExtractor:
    """Get singleton entity extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor
