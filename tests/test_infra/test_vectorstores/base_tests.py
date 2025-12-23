"""Base test class for unified vector store testing.

This module provides abstract test classes that can be used to test any vector store
implementation with the same test suite, ensuring consistency across all backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pytest


class BaseVectorStoreTest(ABC):
    """Abstract base class for vector store tests.
    
    Subclass this to create unified tests for any vector store implementation.
    Each vector store (Milvus, Qdrant, ChromaDB) should implement the fixtures
    and this class will provide consistent tests for all.
    """
    
    @abstractmethod
    @pytest.fixture
    def vector_store_service(self) -> Any:
        """Return configured vector store service instance.
        
        Returns:
            Configured service (MilvusService, QdrantService, ChromaService, etc.)
        """
        pass
    
    @abstractmethod
    @pytest.fixture
    def collection_name(self) -> str:
        """Return test collection name.
        
        Returns:
            Name of the collection to use for testing
        """
        pass
    
    @abstractmethod
    @pytest.fixture
    def vector_size(self) -> int:
        """Return vector dimensionality.
        
        Returns:
            Size of vectors (e.g., 384, 768, 1536)
        """
        pass
    
    @pytest.fixture
    def sample_vectors(self, vector_size: int) -> List[List[float]]:
        """Generate sample vectors for testing.
        
        Args:
            vector_size: Dimensionality of vectors
            
        Returns:
            List of sample vectors
        """
        import random
        random.seed(42)
        return [
            [random.random() for _ in range(vector_size)]
            for _ in range(5)
        ]
    
    @pytest.fixture
    def sample_ids(self) -> List[str]:
        """Generate sample document IDs.
        
        Returns:
            List of document IDs
        """
        return [f"doc-{i}" for i in range(5)]
    
    @pytest.fixture
    def sample_metadata(self) -> List[Dict[str, Any]]:
        """Generate sample metadata.
        
        Returns:
            List of metadata dictionaries
        """
        return [
            {"source": "test", "page": i, "category": "A" if i % 2 == 0 else "B"}
            for i in range(5)
        ]
    
    @pytest.fixture
    def sample_documents(self) -> List[str]:
        """Generate sample documents.
        
        Returns:
            List of document texts
        """
        return [
            f"This is test document number {i}"
            for i in range(5)
        ]


class VectorStoreCollectionTests(BaseVectorStoreTest):
    """Unified tests for collection management."""
    
    def test_create_collection(
        self,
        vector_store_service: Any,
        collection_name: str,
        vector_size: int,
    ) -> None:
        """Test creating a collection."""
        # This is an abstract test - subclasses implement specifics
        pass
    
    def test_collection_exists(
        self,
        vector_store_service: Any,
        collection_name: str,
    ) -> None:
        """Test checking if collection exists."""
        pass
    
    def test_drop_collection(
        self,
        vector_store_service: Any,
        collection_name: str,
    ) -> None:
        """Test dropping a collection."""
        pass
    
    def test_list_collections(
        self,
        vector_store_service: Any,
    ) -> None:
        """Test listing collections."""
        pass


class VectorStoreDataTests(BaseVectorStoreTest):
    """Unified tests for data operations."""
    
    def test_add_vectors(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_ids: List[str],
        sample_vectors: List[List[float]],
        sample_metadata: List[Dict[str, Any]],
    ) -> None:
        """Test adding vectors to collection."""
        pass
    
    def test_upsert_vectors(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_ids: List[str],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test upserting vectors."""
        pass
    
    def test_search_vectors(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_vectors: List[List[float]],
    ) -> None:
        """Test searching for similar vectors."""
        pass
    
    def test_get_by_ids(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_ids: List[str],
    ) -> None:
        """Test retrieving vectors by IDs."""
        pass
    
    def test_delete_vectors(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_ids: List[str],
    ) -> None:
        """Test deleting vectors."""
        pass
    
    def test_count_vectors(
        self,
        vector_store_service: Any,
        collection_name: str,
    ) -> None:
        """Test counting vectors in collection."""
        pass


class VectorStoreFilterTests(BaseVectorStoreTest):
    """Unified tests for filtering capabilities."""
    
    def test_search_with_metadata_filter(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_vectors: List[List[float]],
    ) -> None:
        """Test searching with metadata filters."""
        pass
    
    def test_search_with_multiple_filters(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_vectors: List[List[float]],
    ) -> None:
        """Test searching with combined filters."""
        pass


class VectorStoreBatchTests(BaseVectorStoreTest):
    """Unified tests for batch operations."""
    
    def test_batch_add(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_ids: List[str],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test batch adding vectors."""
        pass
    
    def test_batch_delete(
        self,
        vector_store_service: Any,
        collection_name: str,
        sample_ids: List[str],
    ) -> None:
        """Test batch deleting vectors."""
        pass


class VectorStoreHealthTests(BaseVectorStoreTest):
    """Unified tests for health and status checks."""
    
    def test_health_check(
        self,
        vector_store_service: Any,
    ) -> None:
        """Test health check."""
        assert vector_store_service.health_check() is True
    
    def test_connection(
        self,
        vector_store_service: Any,
    ) -> None:
        """Test connection is established."""
        pass


__all__ = [
    "BaseVectorStoreTest",
    "VectorStoreCollectionTests",
    "VectorStoreDataTests",
    "VectorStoreFilterTests",
    "VectorStoreBatchTests",
    "VectorStoreHealthTests",
]
