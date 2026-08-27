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

    @property
    def pointer(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


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
    added_files: int = 0
    changed_files: int = 0
    unchanged_files: int = 0
    deleted_files: int = 0


class AnswerBundle(BaseModel):
    question: str
    hits: list[RetrievalHit]

    @property
    def evidence(self) -> list[SourceRef]:
        return [hit.chunk.source for hit in self.hits]


class Citation(BaseModel):
    path: str
    start_line: int
    end_line: int
    symbol: str | None = None
    score: float

    @property
    def pointer(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class GroundedAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)


class QueryPlan(BaseModel):
    original: str
    subqueries: list[str]
    tokens: list[str] = Field(default_factory=list)


class DecisionRecord(BaseModel):
    source: Literal["git", "adr"]
    title: str
    reference: str
    summary: str = ""
    timestamp: str | None = None


class DependencyEdge(BaseModel):
    source: str
    target: str
    import_name: str
    resolved: bool = True


class ImpactedFile(BaseModel):
    path: str
    depth: int = Field(ge=0)
    reason: str


class ImpactReport(BaseModel):
    changed_files: list[str]
    impacted_files: list[ImpactedFile]
    edges_considered: int


class ContextPack(BaseModel):
    question: str
    plan: QueryPlan
    answer: GroundedAnswer
    decisions: list[DecisionRecord] = Field(default_factory=list)
    impact: ImpactReport | None = None


class EvidenceIntegrityReport(BaseModel):
    citations: int
    unique_sources: int
    valid_pointers: int
    evidence_integrity: float = Field(ge=0.0, le=1.0)


class EvalCase(BaseModel):
    question: str
    expected_paths: list[str] = Field(default_factory=list)
    expected_symbols: list[str] = Field(default_factory=list)


class EvalCaseResult(BaseModel):
    question: str
    passed: bool
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    matched_path: str | None = None
    matched_symbol: str | None = None


class EvalReport(BaseModel):
    cases: int
    hit_rate_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    results: list[EvalCaseResult]


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
