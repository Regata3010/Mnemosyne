"""
Importance Scorer - determines if a memory is worth storing.

Uses a multi-signal approach:
1. Sentiment intensity (VADER) - strong emotions = important
2. Entity density (spaCy) - names, dates, amounts = important  
3. Actionability - requests, commitments, deadlines = important
4. Novelty heuristics - new info patterns

Target: <5ms per scoring (quantize if needed)
"""

import re
from dataclasses import dataclass
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class ImportanceSignals:
    """Individual signals that contribute to importance score."""
    sentiment_intensity: float  # 0-1, how emotionally charged
    entity_density: float       # 0-1, named entities per word
    actionability: float        # 0-1, contains actionable items
    specificity: float          # 0-1, specific vs generic
    
    # Raw values for debugging
    raw_sentiment: dict[str, float] | None = None
    entities_found: list[str] | None = None
    action_keywords_found: list[str] | None = None


class ImportanceScorer:
    """
    Scores memory importance using heuristic signals.
    
    Design: Fast heuristics first, LLM fallback for edge cases.
    Target latency: <5ms for heuristic path.
    """
    
    # Actionability keywords - requests, commitments, deadlines
    ACTION_KEYWORDS = {
        # Requests
        'please', 'need', 'want', 'require', 'must', 'should',
        'asap', 'urgent', 'immediately', 'priority',
        # Commitments
        'promise', 'guarantee', 'commit', 'agree', 'will do',
        'scheduled', 'appointment', 'deadline', 'due',
        # Complaints/escalation
        'complain', 'complaint', 'frustrated', 'angry', 'upset',
        'cancel', 'refund', 'escalate', 'manager', 'supervisor',
        # Important events
        'bought', 'purchased', 'ordered', 'paid', 'charged',
        'broken', 'damaged', 'missing', 'wrong', 'error',
        # Customer service actions
        'change', 'update', 'modify', 'edit', 'delete', 'remove',
        'track', 'status', 'shipping', 'delivery', 'return',
        'account', 'password', 'login', 'access', 'subscription',
        'help', 'support', 'issue', 'problem', 'trouble',
    }
    
    # High-value entity patterns (regex)
    ENTITY_PATTERNS = [
        r'\$[\d,]+(?:\.\d{2})?',           # Money: $100, $1,000.00
        r'\d{1,2}/\d{1,2}/\d{2,4}',         # Dates: 1/15/2024
        r'\d{4}-\d{2}-\d{2}',               # ISO dates: 2024-01-15
        r'#\d+',                            # Order/ticket numbers: #12345
        r'\b[A-Z]{2,}\d+\b',                # IDs: ABC123, ORDER456
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',   # Phone numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
        r'\{\{[^}]+\}\}',                   # Template vars: {{Order Number}}
        r'\b(?:order|account|ticket|case|ref)[\s#-]*\d+\b',  # order 12345, account-789
    ]
    
    # Generic/low-value phrases
    GENERIC_PHRASES = {
        'hi', 'hello', 'hey', 'good morning', 'good afternoon',
        'how are you', "how's it going", 'thanks', 'thank you',
        'bye', 'goodbye', 'have a nice day', 'take care',
        'okay', 'ok', 'sure', 'yes', 'no', 'maybe',
        'i see', 'got it', 'understood', 'makes sense',
    }
    
    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        self._entity_patterns = [re.compile(p, re.IGNORECASE) for p in self.ENTITY_PATTERNS]
        self._spacy_nlp = None  # Lazy load
    
    @property
    def spacy_nlp(self):
        """Lazy load spaCy model."""
        if self._spacy_nlp is None:
            import spacy
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed, use blank
                self._spacy_nlp = spacy.blank("en")
        return self._spacy_nlp
    
    def score(self, content: str, context: dict[str, Any] | None = None) -> tuple[float, ImportanceSignals]:
        """
        Score the importance of content.
        
        Args:
            content: The text to score
            context: Optional context (user_id, session history, etc.)
        
        Returns:
            (importance_score, signals) where score is 0-1
        """
        content_lower = content.lower().strip()
        words = content.split()
        word_count = len(words)
        
        # Fast path: reject obvious low-value content
        if self._is_generic(content_lower):
            return 0.1, ImportanceSignals(
                sentiment_intensity=0.0,
                entity_density=0.0,
                actionability=0.0,
                specificity=0.0,
            )
        
        # Signal 1: Sentiment intensity
        sentiment_score, raw_sentiment = self._score_sentiment(content)
        
        # Signal 2: Entity density
        entity_score, entities = self._score_entities(content, word_count)
        
        # Signal 3: Actionability
        action_score, action_keywords = self._score_actionability(content_lower)
        
        # Signal 4: Specificity
        specificity_score = self._score_specificity(content, word_count)
        
        # Combine signals with weights
        weights = {
            'sentiment': 0.25,
            'entities': 0.30,
            'actionability': 0.30,
            'specificity': 0.15,
        }
        
        combined = (
            weights['sentiment'] * sentiment_score +
            weights['entities'] * entity_score +
            weights['actionability'] * action_score +
            weights['specificity'] * specificity_score
        )
        
        # Normalize to 0-1 range
        importance = min(max(combined, 0.0), 1.0)
        
        signals = ImportanceSignals(
            sentiment_intensity=sentiment_score,
            entity_density=entity_score,
            actionability=action_score,
            specificity=specificity_score,
            raw_sentiment=raw_sentiment,
            entities_found=entities,
            action_keywords_found=action_keywords,
        )
        
        return importance, signals
    
    def _is_generic(self, content_lower: str) -> bool:
        """Check if content is generic/low-value."""
        # Exact match for very short generic phrases
        if content_lower in self.GENERIC_PHRASES:
            return True
        
        # Check if content is mostly generic
        words = content_lower.split()
        if len(words) <= 3:
            generic_count = sum(1 for w in words if w in self.GENERIC_PHRASES)
            if generic_count >= len(words) - 1:
                return True
        
        return False
    
    def _score_sentiment(self, content: str) -> tuple[float, dict[str, float]]:
        """
        Score based on sentiment intensity.
        
        Strong positive or negative = important.
        Neutral = less important.
        """
        scores = self._vader.polarity_scores(content)
        
        # Compound score is -1 to 1, we want intensity (absolute value)
        compound = scores['compound']
        intensity = abs(compound)
        
        # Also consider if there's strong pos/neg without neutralizing
        pos_neg_intensity = max(scores['pos'], scores['neg'])
        
        # Combine: high compound OR strong pos/neg
        sentiment_score = max(intensity, pos_neg_intensity)
        
        return sentiment_score, scores
    
    def _score_entities(self, content: str, word_count: int) -> tuple[float, list[str]]:
        """
        Score based on named entity density.
        
        More entities relative to length = more specific = more important.
        """
        entities = []
        
        # Regex-based entity extraction (fast)
        for pattern in self._entity_patterns:
            matches = pattern.findall(content)
            entities.extend(matches)
        
        # spaCy NER (slower but more comprehensive)
        try:
            doc = self.spacy_nlp(content)
            for ent in doc.ents:
                if ent.label_ in {'PERSON', 'ORG', 'GPE', 'MONEY', 'DATE', 'TIME', 'PRODUCT'}:
                    entities.append(f"{ent.label_}:{ent.text}")
        except Exception:
            pass  # spaCy failed, continue with regex entities
        
        # Score: entities per word, capped
        if word_count == 0:
            return 0.0, entities
        
        density = len(entities) / word_count
        # Cap at 0.5 entities per word for max score
        entity_score = min(density / 0.5, 1.0)
        
        return entity_score, entities
    
    def _score_actionability(self, content_lower: str) -> tuple[float, list[str]]:
        """
        Score based on actionable keywords.
        
        Requests, commitments, complaints = important.
        """
        found_keywords = []
        
        for keyword in self.ACTION_KEYWORDS:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        # More keywords = more actionable, but diminishing returns
        if not found_keywords:
            return 0.0, found_keywords
        
        # 1 keyword = 0.4, 2 = 0.6, 3+ = 0.8+
        score = min(0.3 + (len(found_keywords) * 0.2), 1.0)
        
        return score, found_keywords
    
    def _score_specificity(self, content: str, word_count: int) -> float:
        """
        Score based on specificity heuristics.
        
        Longer, more detailed content tends to be more specific.
        """
        if word_count == 0:
            return 0.0
        
        # Length score: 10 words = 0.5, 30+ words = 1.0
        length_score = min(word_count / 30, 1.0)
        
        # Punctuation density (questions, emphasis)
        punct_count = sum(1 for c in content if c in '?!:;')
        punct_score = min(punct_count / 3, 1.0)
        
        # Number density (specific quantities)
        number_count = len(re.findall(r'\d+', content))
        number_score = min(number_count / 3, 1.0)
        
        # Combine
        specificity = (length_score * 0.4 + punct_score * 0.3 + number_score * 0.3)
        
        return specificity


# Singleton instance for reuse
_scorer: ImportanceScorer | None = None


def get_importance_scorer() -> ImportanceScorer:
    """Get the singleton importance scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ImportanceScorer()
    return _scorer
