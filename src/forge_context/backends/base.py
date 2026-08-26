from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import IndexedChunk


class VectorBackend(ABC):
    @abstractmethod
    def replace(self, chunks: list[IndexedChunk]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], limit: int = 8) -> list[tuple[IndexedChunk, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def all(self) -> list[IndexedChunk]: ...
