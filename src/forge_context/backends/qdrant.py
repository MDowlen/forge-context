from __future__ import annotations

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

    def replace(self, chunks: list[IndexedChunk]) -> None:
        self._ensure()
        self.client.delete_collection(self.collection)
        self._ensure()
        points = [
            PointStruct(
                id=idx + 1,
                vector=item.vector,
                payload={"chunk": item.model_dump(mode="json", exclude={"vector"})},
            )
            for idx, item in enumerate(chunks)
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
        ).points
        output: list[tuple[IndexedChunk, float]] = []
        for hit in result:
            chunk = ContextChunk.model_validate(hit.payload["chunk"])
            output.append((IndexedChunk(**chunk.model_dump(), vector=vector), float(hit.score)))
        return output

    def count(self) -> int:
        self._ensure()
        return int(self.client.count(self.collection, exact=True).count)
