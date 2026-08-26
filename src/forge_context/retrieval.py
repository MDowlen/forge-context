from __future__ import annotations

import re

from .backends.base import VectorBackend
from .embeddings import EmbeddingProvider
from .models import AnswerBundle, Citation, ContextChunk, GroundedAnswer, RetrievalHit


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
    def __init__(self, backend: VectorBackend, embedder: EmbeddingProvider) -> None:
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

    def grounded_answer(self, question: str, limit: int = 6) -> GroundedAnswer:
        bundle = self.ask(question, limit=limit)
        if not bundle.hits:
            return GroundedAnswer(
                question=question,
                answer="No grounded repository evidence was found.",
                citations=[],
                confidence=0.0,
            )

        best = bundle.hits[0]
        source = best.chunk.source
        symbol_text = f" in `{source.symbol}`" if source.symbol else ""
        answer = (
            f"Best grounded match: `{source.path}` lines {source.start_line}-{source.end_line}"
            f"{symbol_text}. Review the cited evidence before taking an automated action."
        )
        citations = [
            Citation(
                path=hit.chunk.source.path,
                start_line=hit.chunk.source.start_line,
                end_line=hit.chunk.source.end_line,
                symbol=hit.chunk.source.symbol,
                score=max(0.0, min(1.0, hit.final_score)),
            )
            for hit in bundle.hits
        ]
        confidence = max(0.0, min(1.0, best.final_score))
        return GroundedAnswer(
            question=question,
            answer=answer,
            citations=citations,
            confidence=confidence,
        )
