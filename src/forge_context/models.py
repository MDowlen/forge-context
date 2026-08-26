from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    symbol: str | None = None
    language: str | None = None
    content_sha256: str


class ContextChunk(BaseModel):
    id: str
    text: str
    source: SourceRef
    kind: Literal["code", "document", "config", "unknown"] = "unknown"
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class IndexedChunk(ContextChunk):
    vector: list[float]


class RetrievalHit(BaseModel):
    chunk: ContextChunk
    vector_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0


class SyncReport(BaseModel):
    root: str
    files_seen: int
    files_indexed: int
    chunks_indexed: int
    skipped_files: int
    languages: dict[str, int]


class AnswerBundle(BaseModel):
    question: str
    hits: list[RetrievalHit]

    @property
    def evidence(self) -> list[SourceRef]:
        return [hit.chunk.source for hit in self.hits]


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
