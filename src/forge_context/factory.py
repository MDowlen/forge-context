from __future__ import annotations

from .backends.base import VectorBackend
from .backends.local import LocalJsonBackend
from .backends.qdrant import QdrantBackend
from .config import Settings


def make_backend(settings: Settings) -> VectorBackend:
    if settings.qdrant_url:
        return QdrantBackend(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            dimensions=settings.embedding_dimensions,
        )
    return LocalJsonBackend(settings.state_dir)
