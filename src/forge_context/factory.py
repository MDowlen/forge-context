from __future__ import annotations

from .backends.base import VectorBackend
from .backends.local import LocalJsonBackend
from .backends.qdrant import QdrantBackend
from .config import Settings
from .embeddings import (
    EmbeddingProvider,
    HashEmbedding,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
)


def make_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedding(settings.embedding_model or "all-MiniLM-L6-v2")
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when FORGE_EMBEDDING_PROVIDER=openai")
        return OpenAIEmbedding(
            api_key=settings.openai_api_key,
            model=settings.embedding_model or "text-embedding-3-small",
            dimensions=settings.embedding_dimensions,
            base_url=settings.openai_base_url,
        )
    return HashEmbedding(settings.embedding_dimensions)


def make_backend(settings: Settings, dimensions: int | None = None) -> VectorBackend:
    active_dimensions = dimensions or settings.embedding_dimensions
    if settings.qdrant_url:
        return QdrantBackend(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            dimensions=active_dimensions,
        )
    return LocalJsonBackend(settings.state_dir)
