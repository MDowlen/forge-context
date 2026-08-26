from __future__ import annotations

import re

from .backends.base import VectorBackend
from .embeddings import HashEmbedding
from .models import AnswerBundle, ContextChunk, RetrievalHit


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")


def lexical_score(question: str, chunk: ContextChunk) -> float:
    q = set(TOKEN_RE.findall(question.lower()))
    if not q:
        return 0.0
    text = set(TOKEN_RE.findall(chunk.text.lower()))
    path = set(TOKEN_RE.findall(chunk.source.path.lower()))
    symbol = set(TOKEN_RE.findall((chunk.source.symbol or "").lower()))
    overlap = len(q & text)
    bonus = 2 * len(q & path) + 3 * len(q & symbol)
    return (overlap + bonus) / max(1, len(q) * 2)


class Retriever:
    def __init__(self, backend: VectorBackend, embedder: HashEmbedding) -> None:
        self.backend = backend
        self.embedder = embedder

    def ask(self, question: str, limit: int = 6) -> AnswerBundle:
        vector = self.embedder.embed(question)
        candidates = self.backend.search(vector, limit=max(limit * 3, 12))
        hits: list[RetrievalHit] = []
        for chunk, vector_score in candidates:
            lexical = lexical_score(question, chunk)
            final = 0.65 * vector_score + 0.35 * lexical
            hits.append(
                RetrievalHit(
                    chunk=ContextChunk.model_validate(chunk.model_dump(exclude={"vector"})),
                    vector_score=vector_score,
                    lexical_score=lexical,
                    final_score=final,
                )
            )
        hits.sort(key=lambda hit: hit.final_score, reverse=True)
        return AnswerBundle(question=question, hits=hits[:limit])
