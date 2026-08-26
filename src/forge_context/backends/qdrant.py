from __future__ import annotations

import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..models import ContextChunk, IndexedChunk
from .base import VectorBackend


class QdrantBackend(VectorBackend):
    def __init__(self, url: str, api_key: str | None, collection: str, dimensions: int) -> None:
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.dimensions = dimensions

    def _ensure(self) -> None:
        names = {item.name for item in self.client.get_collections().collections}
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> int:
        digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") & ((1 << 63) - 1)

    def replace(self, chunks: list[IndexedChunk]) -> None:
        self._ensure()
        self.client.delete_collection(self.collection)
        self._ensure()
        points = [
            PointStruct(
                id=self._point_id(item.id),
                vector=item.vector,
                payload={"chunk": item.model_dump(mode="json", exclude={"vector"})},
            )
            for item in chunks
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int = 8) -> list[tuple[IndexedChunk, float]]:
        self._ensure()
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        ).points
        output: list[tuple[IndexedChunk, float]] = []
        for hit in result:
            chunk = ContextChunk.model_validate(hit.payload["chunk"])
            stored_vector = list(hit.vector) if isinstance(hit.vector, list) else vector
            output.append((IndexedChunk(**chunk.model_dump(), vector=stored_vector), float(hit.score)))
        return output

    def count(self) -> int:
        self._ensure()
        return int(self.client.count(self.collection, exact=True).count)

    def all(self) -> list[IndexedChunk]:
        self._ensure()
        output: list[IndexedChunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                chunk = ContextChunk.model_validate(point.payload["chunk"])
                vector = list(point.vector) if isinstance(point.vector, list) else []
                output.append(IndexedChunk(**chunk.model_dump(), vector=vector))
            if offset is None:
                break
        return output
