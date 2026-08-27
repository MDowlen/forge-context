from __future__ import annotations

import re
from collections import Counter

from .models import QueryPlan, RetrievalHit

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
SPLIT_RE = re.compile(r"\b(?:and|or|then|also|plus)\b|[?;]", re.IGNORECASE)


def plan_query(question: str) -> QueryPlan:
    cleaned = " ".join(question.strip().split())
    pieces = [piece.strip(" ,.") for piece in SPLIT_RE.split(cleaned) if piece.strip(" ,.")]
    subqueries = [cleaned]
    for piece in pieces:
        if piece.lower() != cleaned.lower() and len(piece) >= 4:
            subqueries.append(piece)
    tokens = sorted(set(TOKEN_RE.findall(cleaned.lower())))
    return QueryPlan(original=cleaned, subqueries=list(dict.fromkeys(subqueries)), tokens=tokens)


def rerank_diverse(hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
    """Favor strong evidence while avoiding a top-K made of one file only."""
    selected: list[RetrievalHit] = []
    path_counts: Counter[str] = Counter()
    remaining = list(hits)

    while remaining and len(selected) < limit:
        best_index = 0
        best_score = float("-inf")
        for index, hit in enumerate(remaining):
            path = hit.chunk.source.path
            diversity_penalty = min(path_counts[path] * 0.08, 0.24)
            symbol_bonus = 0.03 if hit.chunk.source.symbol else 0.0
            score = hit.final_score - diversity_penalty + symbol_bonus
            if score > best_score:
                best_score = score
                best_index = index
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        path_counts[chosen.chunk.source.path] += 1

    return selected
