"""
Memory Consolidation System.

Implements memory lifecycle management:
1. Merge similar memories over time
2. Decay/forget based on access patterns
3. Compress memories that are rarely accessed

Inspired by human memory consolidation where:
- Short-term → long-term through rehearsal
- Similar memories merge into gist memories
- Unused memories decay unless reinforced
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class ConsolidationStrategy(Enum):
    """How to handle similar memories."""
    MERGE = "merge"           # Combine into single gist memory
    COMPRESS = "compress"     # Keep but reduce detail
    ARCHIVE = "archive"       # Move to cold storage
    FORGET = "forget"         # Delete entirely


@dataclass
class MemoryCluster:
    """A cluster of similar memories."""
    cluster_id: str
    memories: List[Dict[str, Any]]
    centroid: np.ndarray
    avg_importance: float
    avg_age_days: float
    total_access_count: int
    
    @property
    def size(self) -> int:
        return len(self.memories)


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""
    
    # Similarity threshold for merging (cosine similarity)
    merge_threshold: float = 0.85
    
    # Time windows
    consolidation_window_hours: int = 24  # Run consolidation daily
    decay_start_days: int = 7            # Start decaying after 7 days
    archive_after_days: int = 30         # Archive after 30 days
    forget_after_days: int = 90          # Forget after 90 days (if never accessed)
    
    # Access-based retention
    access_boost_days: float = 7.0       # Each access adds N days to retention
    min_importance_to_keep: float = 0.2  # Below this, subject to faster decay
    
    # Merge settings
    min_cluster_size: int = 2            # Minimum memories to form a cluster
    max_cluster_size: int = 10           # Maximum memories per cluster
    
    # Compression
    compress_after_days: int = 14        # Compress details after N days
    summary_max_length: int = 200        # Max length for compressed summary


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""
    started_at: datetime
    completed_at: datetime
    memories_processed: int
    clusters_formed: int
    memories_merged: int
    memories_compressed: int
    memories_archived: int
    memories_forgotten: int
    
    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


class MemoryConsolidator:
    """
    Handles memory lifecycle management.
    
    Core idea: Memories aren't permanent. They should:
    - Merge when highly similar (form "gist" memories)
    - Decay when not accessed
    - Be reinforced when accessed
    """
    
    def __init__(self, config: ConsolidationConfig = None):
        self.config = config or ConsolidationConfig()
        self._embedder = None
    
    def _get_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            try:
                from src.core.embeddings import get_embedder
                self._embedder = get_embedder()
            except ImportError:
                self._embedder = None
        return self._embedder
    
    def compute_decay_score(
        self,
        memory: Dict[str, Any],
        current_time: datetime = None,
    ) -> float:
        """
        Compute decay score for a memory (0 = should forget, 1 = fully retained).
        
        Factors:
        - Time since creation (older = more decay)
        - Access count (more accesses = less decay)
        - Importance score (higher importance = less decay)
        """
        current_time = current_time or datetime.utcnow()
        created_at = memory.get("created_at", current_time)
        
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        age_days = (current_time - created_at).total_seconds() / 86400
        access_count = memory.get("access_count", 0)
        importance = memory.get("importance_score", 0.5)
        last_accessed = memory.get("last_accessed_at", created_at)
        
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
        
        days_since_access = (current_time - last_accessed).total_seconds() / 86400
        
        # Base decay (exponential)
        # Half-life depends on importance
        half_life = self.config.decay_start_days * (1 + importance)
        base_decay = 0.5 ** (days_since_access / half_life)
        
        # Access boost (each access extends retention)
        access_boost = min(1.0, access_count * 0.1)  # Max 100% boost
        
        # Importance boost
        importance_boost = importance * 0.3
        
        # Combined score
        decay_score = base_decay * (1 + access_boost + importance_boost)
        
        return min(1.0, max(0.0, decay_score))
    
    def should_forget(self, memory: Dict[str, Any]) -> bool:
        """Check if memory should be forgotten."""
        decay_score = self.compute_decay_score(memory)
        importance = memory.get("importance_score", 0.5)
        access_count = memory.get("access_count", 0)
        
        # Never forget high-importance memories that have been accessed
        if importance > 0.7 and access_count > 0:
            return False
        
        # Forget if decay score is very low and importance is low
        if decay_score < 0.1 and importance < self.config.min_importance_to_keep:
            return True
        
        # Forget if never accessed and very old
        created_at = memory.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        age_days = (datetime.utcnow() - created_at).total_seconds() / 86400
        
        if access_count == 0 and age_days > self.config.forget_after_days:
            return True
        
        return False
    
    def should_archive(self, memory: Dict[str, Any]) -> bool:
        """Check if memory should be archived (moved to cold storage)."""
        created_at = memory.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        age_days = (datetime.utcnow() - created_at).total_seconds() / 86400
        access_count = memory.get("access_count", 0)
        
        # Archive if old and rarely accessed
        if age_days > self.config.archive_after_days:
            if access_count < 3:  # Rarely accessed
                return True
        
        return False
    
    def find_similar_clusters(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[MemoryCluster]:
        """
        Find clusters of similar memories that could be merged.
        
        Uses embeddings for semantic similarity.
        """
        if len(memories) < self.config.min_cluster_size:
            return []
        
        embedder = self._get_embedder()
        
        # Get embeddings
        embeddings = []
        for memory in memories:
            if "embedding" in memory and memory["embedding"] is not None:
                embeddings.append(np.array(memory["embedding"]))
            elif embedder:
                emb = embedder.embed(memory.get("content", ""))
                embeddings.append(emb)
            else:
                # Random embedding as fallback
                embeddings.append(np.random.randn(384).astype(np.float32))
        
        embeddings = np.array(embeddings)
        
        # Compute pairwise similarities
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        similarities = np.dot(normalized, normalized.T)
        
        # Greedy clustering
        clusters = []
        assigned = set()
        
        for i in range(len(memories)):
            if i in assigned:
                continue
            
            # Find similar memories
            similar_indices = [i]
            for j in range(i + 1, len(memories)):
                if j in assigned:
                    continue
                if similarities[i, j] >= self.config.merge_threshold:
                    similar_indices.append(j)
                    if len(similar_indices) >= self.config.max_cluster_size:
                        break
            
            if len(similar_indices) >= self.config.min_cluster_size:
                # Form cluster
                cluster_memories = [memories[idx] for idx in similar_indices]
                cluster_embeddings = embeddings[similar_indices]
                centroid = cluster_embeddings.mean(axis=0)
                
                avg_importance = np.mean([
                    m.get("importance_score", 0.5) for m in cluster_memories
                ])
                
                avg_age = np.mean([
                    (datetime.utcnow() - (
                        datetime.fromisoformat(m["created_at"])
                        if isinstance(m.get("created_at"), str)
                        else m.get("created_at", datetime.utcnow())
                    )).total_seconds() / 86400
                    for m in cluster_memories
                ])
                
                total_access = sum(m.get("access_count", 0) for m in cluster_memories)
                
                cluster = MemoryCluster(
                    cluster_id=str(uuid4()),
                    memories=cluster_memories,
                    centroid=centroid,
                    avg_importance=avg_importance,
                    avg_age_days=avg_age,
                    total_access_count=total_access,
                )
                clusters.append(cluster)
                
                assigned.update(similar_indices)
        
        return clusters
    
    def merge_cluster(self, cluster: MemoryCluster) -> Dict[str, Any]:
        """
        Merge a cluster of similar memories into a single "gist" memory.
        
        Combines content, preserves highest importance, aggregates metadata.
        """
        # Extract contents
        contents = [m.get("content", "") for m in cluster.memories]
        
        # Create merged content (simple concatenation with dedup)
        # In production, you'd use an LLM to summarize
        unique_sentences = []
        seen = set()
        for content in contents:
            for sentence in content.split(". "):
                normalized = sentence.lower().strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    unique_sentences.append(sentence.strip())
        
        merged_content = ". ".join(unique_sentences[:5])  # Keep top 5 unique sentences
        if len(merged_content) > 500:
            merged_content = merged_content[:500] + "..."
        
        # Get highest importance
        max_importance = max(m.get("importance_score", 0.5) for m in cluster.memories)
        
        # Combine entities
        all_entities = []
        for m in cluster.memories:
            all_entities.extend(m.get("entities", []))
        unique_entities = list({str(e): e for e in all_entities}.values())
        
        # Combine metadata
        merged_metadata = {
            "merged_from": [m.get("id") for m in cluster.memories],
            "merge_count": len(cluster.memories),
            "original_categories": list(set(
                m.get("metadata", {}).get("category", "unknown")
                for m in cluster.memories
            )),
        }
        
        # Get oldest creation date and newest access
        created_dates = [
            datetime.fromisoformat(m["created_at"])
            if isinstance(m.get("created_at"), str)
            else m.get("created_at", datetime.utcnow())
            for m in cluster.memories
        ]
        
        return {
            "id": cluster.cluster_id,
            "content": merged_content,
            "importance_score": max_importance,
            "entities": unique_entities,
            "embedding": cluster.centroid,
            "metadata": merged_metadata,
            "created_at": min(created_dates),
            "access_count": cluster.total_access_count,
            "is_merged": True,
        }
    
    def compress_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress a memory by reducing detail while preserving key info.
        
        In production, you'd use an LLM. Here we do simple truncation.
        """
        content = memory.get("content", "")
        
        if len(content) <= self.config.summary_max_length:
            return memory  # Already short enough
        
        # Simple compression: keep first N chars + key entities
        compressed_content = content[:self.config.summary_max_length]
        
        # Append entities as keywords
        entities = memory.get("entities", [])
        if entities:
            entity_str = " | Entities: " + ", ".join(
                str(e.get("value", e)) if isinstance(e, dict) else str(e)
                for e in entities[:5]
            )
            compressed_content = compressed_content[:150] + entity_str
        
        compressed = memory.copy()
        compressed["content"] = compressed_content
        compressed["metadata"] = {
            **memory.get("metadata", {}),
            "compressed": True,
            "original_length": len(content),
        }
        
        return compressed
    
    async def run_consolidation(
        self,
        memories: List[Dict[str, Any]],
        dry_run: bool = False,
    ) -> Tuple[ConsolidationResult, List[Dict[str, Any]]]:
        """
        Run full consolidation pass on memories.
        
        Returns:
            - ConsolidationResult with stats
            - List of consolidated memories (if not dry_run)
        """
        started_at = datetime.utcnow()
        
        result = ConsolidationResult(
            started_at=started_at,
            completed_at=started_at,
            memories_processed=len(memories),
            clusters_formed=0,
            memories_merged=0,
            memories_compressed=0,
            memories_archived=0,
            memories_forgotten=0,
        )
        
        # Phase 1: Mark for forgetting
        to_forget = []
        to_archive = []
        to_keep = []
        
        for memory in memories:
            if self.should_forget(memory):
                to_forget.append(memory)
            elif self.should_archive(memory):
                to_archive.append(memory)
            else:
                to_keep.append(memory)
        
        result.memories_forgotten = len(to_forget)
        result.memories_archived = len(to_archive)
        
        # Phase 2: Find and merge similar memories
        clusters = self.find_similar_clusters(to_keep)
        result.clusters_formed = len(clusters)
        
        merged_ids = set()
        merged_memories = []
        
        for cluster in clusters:
            merged = self.merge_cluster(cluster)
            merged_memories.append(merged)
            result.memories_merged += len(cluster.memories)
            for m in cluster.memories:
                merged_ids.add(m.get("id"))
        
        # Phase 3: Compress old memories
        final_memories = []
        for memory in to_keep:
            if memory.get("id") in merged_ids:
                continue  # Already merged
            
            created_at = memory.get("created_at", datetime.utcnow())
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            
            age_days = (datetime.utcnow() - created_at).total_seconds() / 86400
            
            if age_days > self.config.compress_after_days:
                compressed = self.compress_memory(memory)
                if compressed != memory:
                    result.memories_compressed += 1
                final_memories.append(compressed)
            else:
                final_memories.append(memory)
        
        # Add merged memories
        final_memories.extend(merged_memories)
        
        result.completed_at = datetime.utcnow()
        
        if dry_run:
            return result, memories  # Return original
        
        return result, final_memories


def get_consolidator(config: ConsolidationConfig = None) -> MemoryConsolidator:
    """Get memory consolidator instance."""
    return MemoryConsolidator(config)
