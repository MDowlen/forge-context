from __future__ import annotations

from pathlib import Path

from .backends.base import VectorBackend
from .dependency import analyze_impact
from .embeddings import EmbeddingProvider
from .history import collect_decision_context
from .models import ContextPack
from .query import plan_query
from .retrieval import Retriever


class ContextEngine:
    """High-level ForgeContext interface for CLIs, agents, and PR workflows."""

    def __init__(self, backend: VectorBackend, embedder: EmbeddingProvider) -> None:
        self.retriever = Retriever(backend, embedder)

    def ask(self, question: str, limit: int = 6):
        return self.retriever.grounded_answer(question, limit=limit)

    def context_pack(
        self,
        root: Path,
        question: str,
        *,
        changed_files: list[str] | None = None,
        limit: int = 6,
        decision_limit: int = 12,
        impact_depth: int = 4,
    ) -> ContextPack:
        root = root.resolve()
        plan = plan_query(question)
        answer = self.retriever.grounded_answer(question, limit=limit)
        decisions = collect_decision_context(root, limit=decision_limit)
        impact = None
        if changed_files:
            impact = analyze_impact(root, changed_files, max_depth=impact_depth)
        return ContextPack(
            question=question,
            plan=plan,
            answer=answer,
            decisions=decisions,
            impact=impact,
        )
