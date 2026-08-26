from __future__ import annotations

import json
from pathlib import Path

from .models import EvalCase, EvalCaseResult, EvalReport
from .retrieval import Retriever


def load_eval_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases", [])
    return [EvalCase.model_validate(item) for item in raw]


def evaluate_retrieval(retriever: Retriever, cases: list[EvalCase], k: int = 5) -> EvalReport:
    results: list[EvalCaseResult] = []
    reciprocal_ranks: list[float] = []

    for case in cases:
        bundle = retriever.ask(case.question, limit=k)
        rank = 0
        matched_path = None
        matched_symbol = None
        for index, hit in enumerate(bundle.hits, start=1):
            source = hit.chunk.source
            path_match = not case.expected_paths or source.path in case.expected_paths
            symbol_match = not case.expected_symbols or source.symbol in case.expected_symbols
            if path_match and symbol_match:
                rank = index
                matched_path = source.path
                matched_symbol = source.symbol
                break
        reciprocal_rank = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(reciprocal_rank)
        results.append(
            EvalCaseResult(
                question=case.question,
                passed=bool(rank),
                reciprocal_rank=reciprocal_rank,
                matched_path=matched_path,
                matched_symbol=matched_symbol,
            )
        )

    count = len(results)
    hits = sum(1 for result in results if result.passed)
    return EvalReport(
        cases=count,
        hit_rate_at_k=(hits / count) if count else 0.0,
        mean_reciprocal_rank=(sum(reciprocal_ranks) / count) if count else 0.0,
        results=results,
    )
