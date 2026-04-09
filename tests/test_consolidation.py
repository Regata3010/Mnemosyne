"""Tests for memory consolidation system."""

import pytest
from datetime import datetime, timedelta
import numpy as np

from src.core.consolidation import (
    MemoryConsolidator,
    ConsolidationConfig,
    ConsolidationStrategy,
    MemoryCluster,
)


class TestDecayScoring:
    """Test memory decay calculation."""
    
    def test_fresh_memory_no_decay(self):
        """Fresh memories should have high retention."""
        consolidator = MemoryConsolidator()
        memory = {
            "id": "test-1",
            "content": "Important customer info",
            "created_at": datetime.utcnow(),
            "importance_score": 0.5,
            "access_count": 0,
        }
        
        score = consolidator.compute_decay_score(memory)
        assert score > 0.9, "Fresh memory should have minimal decay"
    
    def test_old_memory_decays(self):
        """Old memories should decay."""
        consolidator = MemoryConsolidator()
        memory = {
            "id": "test-1",
            "content": "Old info",
            "created_at": datetime.utcnow() - timedelta(days=30),
            "last_accessed_at": datetime.utcnow() - timedelta(days=30),
            "importance_score": 0.3,
            "access_count": 0,
        }
        
        score = consolidator.compute_decay_score(memory)
        assert score < 0.5, "Old memory should have significant decay"
    
    def test_accessed_memory_retained(self):
        """Frequently accessed memories should be retained."""
        consolidator = MemoryConsolidator()
        memory = {
            "id": "test-1",
            "content": "Useful info",
            "created_at": datetime.utcnow() - timedelta(days=30),
            "last_accessed_at": datetime.utcnow(),  # Recently accessed
            "importance_score": 0.5,
            "access_count": 10,
        }
        
        score = consolidator.compute_decay_score(memory)
        assert score > 0.7, "Recently accessed memory should be retained"
    
    def test_high_importance_slower_decay(self):
        """High importance memories should decay slower."""
        consolidator = MemoryConsolidator()
        
        low_importance = {
            "id": "test-1",
            "content": "Low importance",
            "created_at": datetime.utcnow() - timedelta(days=14),
            "last_accessed_at": datetime.utcnow() - timedelta(days=14),
            "importance_score": 0.2,
            "access_count": 0,
        }
        
        high_importance = {
            "id": "test-2",
            "content": "High importance",
            "created_at": datetime.utcnow() - timedelta(days=14),
            "last_accessed_at": datetime.utcnow() - timedelta(days=14),
            "importance_score": 0.9,
            "access_count": 0,
        }
        
        low_score = consolidator.compute_decay_score(low_importance)
        high_score = consolidator.compute_decay_score(high_importance)
        
        assert high_score > low_score, "High importance should decay slower"


class TestForgetDecision:
    """Test forget/retain decisions."""
    
    def test_forget_old_unaccessed_low_importance(self):
        """Should forget old, never-accessed, low-importance memories."""
        consolidator = MemoryConsolidator()
        memory = {
            "id": "test-1",
            "content": "Stale info",
            "created_at": datetime.utcnow() - timedelta(days=100),
            "importance_score": 0.1,
            "access_count": 0,
        }
        
        assert consolidator.should_forget(memory) is True
    
    def test_retain_high_importance_accessed(self):
        """Should never forget high-importance accessed memories."""
        consolidator = MemoryConsolidator()
        memory = {
            "id": "test-1",
            "content": "Critical customer issue",
            "created_at": datetime.utcnow() - timedelta(days=100),
            "importance_score": 0.9,
            "access_count": 5,
        }
        
        assert consolidator.should_forget(memory) is False
    
    def test_retain_recent_memory(self):
        """Should retain recent memories."""
        consolidator = MemoryConsolidator()
        memory = {
            "id": "test-1",
            "content": "Recent info",
            "created_at": datetime.utcnow() - timedelta(days=5),
            "importance_score": 0.3,
            "access_count": 0,
        }
        
        assert consolidator.should_forget(memory) is False


class TestArchiveDecision:
    """Test archive decisions."""
    
    def test_archive_old_rarely_accessed(self):
        """Should archive old, rarely accessed memories."""
        consolidator = MemoryConsolidator()
        memory = {
            "id": "test-1",
            "content": "Old but maybe useful",
            "created_at": datetime.utcnow() - timedelta(days=45),
            "access_count": 1,
        }
        
        assert consolidator.should_archive(memory) is True
    
    def test_no_archive_frequently_accessed(self):
        """Should not archive frequently accessed memories."""
        consolidator = MemoryConsolidator()
        memory = {
            "id": "test-1",
            "content": "Still useful",
            "created_at": datetime.utcnow() - timedelta(days=45),
            "access_count": 10,
        }
        
        assert consolidator.should_archive(memory) is False


class TestClustering:
    """Test similar memory clustering."""
    
    def test_cluster_similar_memories(self):
        """Should cluster memories with similar embeddings."""
        consolidator = MemoryConsolidator()
        
        # Create memories with similar content
        # Use very small perturbations to ensure high similarity
        base_embedding = np.random.randn(384).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        memories = []
        for i in range(5):
            # Very small perturbation to ensure cosine similarity > 0.85
            perturbed = base_embedding + np.random.randn(384).astype(np.float32) * 0.01
            perturbed = perturbed / np.linalg.norm(perturbed)  # Normalize
            memories.append({
                "id": f"mem-{i}",
                "content": f"Customer complaint about shipping delay {i}",
                "embedding": perturbed,
                "importance_score": 0.5,
                "created_at": datetime.utcnow(),
            })
        
        # Add a clearly different outlier
        outlier_embedding = -base_embedding  # Opposite direction
        memories.append({
            "id": "outlier",
            "content": "Completely different topic",
            "embedding": outlier_embedding,
            "importance_score": 0.5,
            "created_at": datetime.utcnow(),
        })
        
        clusters = consolidator.find_similar_clusters(memories)
        
        # Should have at least one cluster
        assert len(clusters) >= 1
        
        # Main cluster should contain similar memories
        main_cluster = max(clusters, key=lambda c: c.size)
        assert main_cluster.size >= 3
    
    def test_no_cluster_if_too_few(self):
        """Should not cluster if below minimum size."""
        config = ConsolidationConfig(min_cluster_size=5)
        consolidator = MemoryConsolidator(config)
        
        memories = [
            {"id": "1", "content": "Single memory", "embedding": np.random.randn(384)},
        ]
        
        clusters = consolidator.find_similar_clusters(memories)
        assert len(clusters) == 0


class TestMerging:
    """Test memory merging."""
    
    def test_merge_combines_content(self):
        """Merged memory should combine content from sources."""
        consolidator = MemoryConsolidator()
        
        cluster = MemoryCluster(
            cluster_id="test-cluster",
            memories=[
                {
                    "id": "1",
                    "content": "Customer complained about late delivery.",
                    "importance_score": 0.4,
                    "entities": [{"type": "issue", "value": "late delivery"}],
                    "created_at": datetime.utcnow() - timedelta(days=5),
                },
                {
                    "id": "2",
                    "content": "Package arrived damaged.",
                    "importance_score": 0.6,
                    "entities": [{"type": "issue", "value": "damaged"}],
                    "created_at": datetime.utcnow() - timedelta(days=3),
                },
            ],
            centroid=np.random.randn(384).astype(np.float32),
            avg_importance=0.5,
            avg_age_days=4,
            total_access_count=2,
        )
        
        merged = consolidator.merge_cluster(cluster)
        
        assert "merged_from" in merged["metadata"]
        assert len(merged["metadata"]["merged_from"]) == 2
        assert merged["importance_score"] == 0.6  # Max importance
        assert merged["is_merged"] is True
    
    def test_merge_preserves_highest_importance(self):
        """Merged memory should have highest importance from sources."""
        consolidator = MemoryConsolidator()
        
        cluster = MemoryCluster(
            cluster_id="test-cluster",
            memories=[
                {"id": "1", "content": "A", "importance_score": 0.3, "created_at": datetime.utcnow()},
                {"id": "2", "content": "B", "importance_score": 0.9, "created_at": datetime.utcnow()},
                {"id": "3", "content": "C", "importance_score": 0.5, "created_at": datetime.utcnow()},
            ],
            centroid=np.random.randn(384).astype(np.float32),
            avg_importance=0.57,
            avg_age_days=1,
            total_access_count=0,
        )
        
        merged = consolidator.merge_cluster(cluster)
        assert merged["importance_score"] == 0.9


class TestCompression:
    """Test memory compression."""
    
    def test_compress_long_memory(self):
        """Should compress long memories."""
        config = ConsolidationConfig(summary_max_length=100)
        consolidator = MemoryConsolidator(config)
        
        long_content = "A" * 500
        memory = {
            "id": "1",
            "content": long_content,
            "entities": [],
        }
        
        compressed = consolidator.compress_memory(memory)
        assert len(compressed["content"]) < len(long_content)
        assert compressed["metadata"].get("compressed") is True
    
    def test_no_compress_short_memory(self):
        """Should not compress already short memories."""
        config = ConsolidationConfig(summary_max_length=200)
        consolidator = MemoryConsolidator(config)
        
        memory = {
            "id": "1",
            "content": "Short content",
            "entities": [],
        }
        
        compressed = consolidator.compress_memory(memory)
        assert compressed == memory  # Unchanged


class TestFullConsolidation:
    """Test full consolidation pipeline."""
    
    @pytest.mark.asyncio
    async def test_consolidation_reduces_count(self):
        """Consolidation should reduce memory count through merging."""
        consolidator = MemoryConsolidator()
        
        # Create similar memories with very close embeddings
        base_embedding = np.random.randn(384).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        memories = []
        for i in range(10):
            # Very small perturbation to ensure high similarity (>0.85)
            perturbed = base_embedding + np.random.randn(384).astype(np.float32) * 0.01
            perturbed = perturbed / np.linalg.norm(perturbed)
            memories.append({
                "id": f"mem-{i}",
                "content": f"Customer shipping issue variant {i}",
                "embedding": perturbed,
                "importance_score": 0.5,
                "created_at": datetime.utcnow() - timedelta(days=i),
                "access_count": 1,
            })
        
        result, consolidated = await consolidator.run_consolidation(memories)
        
        assert result.memories_processed == 10
        assert len(consolidated) < 10, "Consolidation should reduce count"
        assert result.clusters_formed >= 1
    
    @pytest.mark.asyncio
    async def test_dry_run_no_changes(self):
        """Dry run should not modify memories."""
        consolidator = MemoryConsolidator()
        
        memories = [
            {
                "id": "1",
                "content": "Test",
                "created_at": datetime.utcnow(),
                "importance_score": 0.5,
            }
        ]
        
        result, returned = await consolidator.run_consolidation(memories, dry_run=True)
        assert returned == memories  # Original returned unchanged
    
    @pytest.mark.asyncio
    async def test_forgets_stale_memories(self):
        """Should forget old, unused, low-importance memories."""
        consolidator = MemoryConsolidator()
        
        memories = [
            {
                "id": "stale",
                "content": "Very old and useless",
                "created_at": datetime.utcnow() - timedelta(days=100),
                "importance_score": 0.1,
                "access_count": 0,
            },
            {
                "id": "fresh",
                "content": "Recent and useful",
                "created_at": datetime.utcnow(),
                "importance_score": 0.5,
                "access_count": 0,
            },
        ]
        
        result, consolidated = await consolidator.run_consolidation(memories)
        
        assert result.memories_forgotten == 1
        assert len(consolidated) == 1
        assert consolidated[0]["id"] == "fresh"
