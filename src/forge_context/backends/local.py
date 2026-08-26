from __future__ import annotations

import json
import math
from pathlib import Path

from ..models import IndexedChunk
from .base import VectorBackend


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        (math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0)
    )


class LocalJsonBackend(VectorBackend):
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.path = state_dir / "index.json"
        self._items: list[IndexedChunk] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._items = [IndexedChunk.model_validate(item) for item in raw]

    def replace(self, chunks: list[IndexedChunk]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._items = list(chunks)
        self.path.write_text(
            json.dumps([item.model_dump(mode="json") for item in chunks], indent=2),
            encoding="utf-8",
        )

    def search(self, vector: list[float], limit: int = 8) -> list[tuple[IndexedChunk, float]]:
        scored = [(item, cosine(vector, item.vector)) for item in self._items]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:limit]

    def count(self) -> int:
        return len(self._items)

    def all(self) -> list[IndexedChunk]:
        return list(self._items)
