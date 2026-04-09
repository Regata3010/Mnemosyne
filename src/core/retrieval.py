"""
Task-Aware Retrieval Engine

Combines vector similarity with task relevance for better memory retrieval.

Components:
1. Vector similarity (pgvector cosine distance)
2. Task context injection (bias toward task-relevant memories)
3. Recency decay (recent memories weighted higher)
4. Optional cross-encoder reranking for precision
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass
class RetrievalConfig:
    """Configuration for retrieval scoring."""
    
    # Weight factors for combined scoring
    similarity_weight: float = 0.4
    task_relevance_weight: float = 0.35
    recency_weight: float = 0.15
    importance_weight: float = 0.10
    
    # Recency decay parameters
    recency_half_life_days: float = 7.0  # Score halves every 7 days
    recency_max_boost: float = 0.3       # Max boost for very recent memories
    
    # Task relevance
    task_boost_factor: float = 1.5       # Multiplier for task-matching memories


class RecencyScorer:
    """
    Scores memories based on recency.
    
    Uses exponential decay: score = base * (0.5 ^ (age_days / half_life))
    """
    
    def __init__(self, half_life_days: float = 7.0, max_boost: float = 0.3):
        self.half_life_days = half_life_days
        self.max_boost = max_boost
    
    def score(self, created_at: datetime, now: datetime | None = None) -> float:
        """
        Calculate recency score for a memory.
        
        Args:
            created_at: When the memory was created
            now: Current time (defaults to utcnow)
        
        Returns:
            Score between 0 and max_boost
        """
        if now is None:
            now = datetime.utcnow()
        
        age = now - created_at
        age_days = age.total_seconds() / (24 * 60 * 60)
        
        # Exponential decay
        decay = math.pow(0.5, age_days / self.half_life_days)
        
        # Scale to max_boost
        return decay * self.max_boost


class TaskRelevanceScorer:
    """
    Scores memories based on relevance to current task context.
    
    Uses keyword matching and semantic overlap.
    """
    
    # Task category keywords
    TASK_CATEGORIES = {
        'billing': {
            'billing', 'payment', 'charge', 'charged', 'invoice', 'refund',
            'credit', 'debit', 'subscription', 'renewal', 'price', 'cost',
            'fee', 'transaction', 'bank', 'card', 'account'
        },
        'shipping': {
            'shipping', 'delivery', 'shipped', 'delivered', 'tracking',
            'package', 'order', 'arrived', 'transit', 'carrier', 'fedex',
            'ups', 'usps', 'delayed', 'lost', 'address'
        },
        'technical': {
            'error', 'bug', 'issue', 'problem', 'broken', 'fix', 'crash',
            'not working', 'failed', 'login', 'password', 'access', 'support',
            'help', 'troubleshoot', 'reset'
        },
        'cancellation': {
            'cancel', 'cancellation', 'terminate', 'end', 'close', 'stop',
            'unsubscribe', 'discontinue', 'quit', 'leave'
        },
        'complaint': {
            'complaint', 'complain', 'frustrated', 'angry', 'upset', 'unhappy',
            'disappointed', 'terrible', 'worst', 'horrible', 'unacceptable',
            'demand', 'escalate', 'manager', 'supervisor'
        },
        'general': {
            'question', 'help', 'information', 'info', 'details', 'explain',
            'how', 'what', 'when', 'where', 'why'
        }
    }
    
    def __init__(self, boost_factor: float = 1.5):
        self.boost_factor = boost_factor
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build reverse index: keyword -> categories."""
        self.keyword_to_categories = {}
        for category, keywords in self.TASK_CATEGORIES.items():
            for keyword in keywords:
                if keyword not in self.keyword_to_categories:
                    self.keyword_to_categories[keyword] = []
                self.keyword_to_categories[keyword].append(category)
    
    def extract_categories(self, text: str) -> set[str]:
        """Extract task categories from text."""
        text_lower = text.lower()
        categories = set()
        
        for keyword, cats in self.keyword_to_categories.items():
            if keyword in text_lower:
                categories.update(cats)
        
        return categories
    
    def score(
        self,
        memory_content: str,
        task_context: str,
        memory_tags: list[str] | None = None,
    ) -> float:
        """
        Score memory relevance to task context.
        
        Args:
            memory_content: The memory text
            task_context: Current task description
            memory_tags: Optional tags on the memory
        
        Returns:
            Relevance score 0-1, higher = more relevant
        """
        # Extract categories from task and memory
        task_categories = self.extract_categories(task_context)
        memory_categories = self.extract_categories(memory_content)
        
        # Add tag-based categories
        if memory_tags:
            for tag in memory_tags:
                memory_categories.update(self.extract_categories(tag))
        
        if not task_categories:
            # No specific task category identified
            return 0.5
        
        # Calculate overlap
        overlap = task_categories & memory_categories
        
        if not overlap:
            return 0.3  # No match, but not zero (could still be relevant)
        
        # More overlap = higher score
        overlap_ratio = len(overlap) / len(task_categories)
        
        # Scale to 0.5-1.0 range (we want matches to boost, not dominate)
        return 0.5 + (overlap_ratio * 0.5)


class HybridRetriever:
    """
    Combines multiple signals for memory retrieval ranking.
    
    Signals:
    1. Vector similarity (from pgvector)
    2. Task relevance (keyword/category matching)
    3. Recency decay (time-based)
    4. Importance score (from ingestion)
    """
    
    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.recency_scorer = RecencyScorer(
            half_life_days=self.config.recency_half_life_days,
            max_boost=self.config.recency_max_boost,
        )
        self.task_scorer = TaskRelevanceScorer(
            boost_factor=self.config.task_boost_factor,
        )
    
    def rerank(
        self,
        results: list[dict[str, Any]],
        task_context: str | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank retrieval results using hybrid scoring.
        
        Args:
            results: List of results with 'memory' and 'similarity_score' keys
            task_context: Optional task description for relevance boosting
            now: Current time for recency calculation
        
        Returns:
            Results sorted by combined relevance_score
        """
        if now is None:
            now = datetime.utcnow()
        
        for result in results:
            memory = result['memory']
            similarity = result.get('similarity_score', 0.5)
            
            # Component scores
            recency = self.recency_scorer.score(memory.created_at, now)
            importance = memory.importance_score
            
            # Task relevance (if context provided)
            if task_context:
                task_relevance = self.task_scorer.score(
                    memory.content,
                    task_context,
                    memory.tags,
                )
            else:
                task_relevance = 0.5  # Neutral
            
            # Combine with weights
            combined = (
                self.config.similarity_weight * similarity +
                self.config.task_relevance_weight * task_relevance +
                self.config.recency_weight * recency +
                self.config.importance_weight * importance
            )
            
            # Normalize to 0-1
            result['relevance_score'] = min(max(combined, 0.0), 1.0)
            
            # Store component scores for debugging
            result['_scoring'] = {
                'similarity': similarity,
                'task_relevance': task_relevance,
                'recency': recency,
                'importance': importance,
                'combined': combined,
            }
        
        # Sort by relevance score descending
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results


# Optional: Cross-encoder reranker for high-precision scenarios
class CrossEncoderReranker:
    """
    Uses a cross-encoder model for precise reranking.
    
    More accurate than bi-encoder similarity but slower.
    Use for top-K refinement when precision matters.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank results using cross-encoder scoring.
        
        Args:
            query: The search query
            results: List of results with 'memory' key
            top_k: Only rerank top K results (for speed)
        
        Returns:
            Results sorted by cross-encoder relevance
        """
        if not results:
            return results
        
        # Optionally limit to top_k for speed
        to_rerank = results[:top_k] if top_k else results
        remaining = results[top_k:] if top_k else []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r['memory'].content) for r in to_rerank]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Update relevance scores
        for result, score in zip(to_rerank, scores):
            result['cross_encoder_score'] = float(score)
            # Blend with existing relevance score
            existing = result.get('relevance_score', 0.5)
            result['relevance_score'] = (existing + float(score)) / 2
        
        # Sort reranked portion
        to_rerank.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return to_rerank + remaining


# Singleton instances
_hybrid_retriever: HybridRetriever | None = None
_cross_encoder: CrossEncoderReranker | None = None


def get_hybrid_retriever(config: RetrievalConfig | None = None) -> HybridRetriever:
    """Get singleton hybrid retriever."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(config)
    return _hybrid_retriever


def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Get singleton cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker()
    return _cross_encoder
