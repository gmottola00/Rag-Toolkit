"""
Core layer: Protocol definitions and type abstractions.

This layer contains only abstract interfaces (Protocols) with zero external dependencies.
All concrete implementations live in the infra layer.

Modules:
    - chunking: Document chunking protocols
    - embedding: Embedding client protocol
    - llm: LLM client protocol
    - vectorstore: Vector store protocol
    - types: Common type definitions
"""

from __future__ import annotations

# Re-export core protocols for easy access
from rag_toolkit.core.chunking import Chunk, TokenChunk
from rag_toolkit.core.embedding import EmbeddingClient
from rag_toolkit.core.llm import LLMClient
from rag_toolkit.core.types import (
    CollectionInfo,
    EmbeddingVector,
    SearchResult,
    VectorMetadata,
)
from rag_toolkit.core.vectorstore import VectorStoreClient

__all__ = [
    # Chunking protocols
    "Chunk",
    "TokenChunk",
    # Client protocols
    "EmbeddingClient",
    "LLMClient",
    "VectorStoreClient",
    # Common types
    "SearchResult",
    "CollectionInfo",
    "VectorMetadata",
    "EmbeddingVector",
]
