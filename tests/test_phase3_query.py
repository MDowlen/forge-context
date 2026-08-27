from forge_context.models import ContextChunk, RetrievalHit, SourceRef
from forge_context.query import plan_query, rerank_diverse


def _hit(path: str, score: float, symbol: str | None = None) -> RetrievalHit:
    chunk = ContextChunk(
        id=f"{path}-{score}",
        text="example code",
        source=SourceRef(
            path=path,
            start_line=1,
            end_line=2,
            symbol=symbol,
            language="python",
            content_sha256="a" * 64,
        ),
        kind="code",
    )
    return RetrievalHit(chunk=chunk, final_score=score)


def test_plan_query_decomposes_compound_question():
    plan = plan_query("Where is retry logic and what calls it?")
    assert plan.original == "Where is retry logic and what calls it?"
    assert len(plan.subqueries) >= 2
    assert "retry" in plan.tokens


def test_rerank_diverse_avoids_single_file_monopoly():
    hits = [
        _hit("a.py", 0.95, "one"),
        _hit("a.py", 0.94, "two"),
        _hit("b.py", 0.91, "three"),
    ]
    ranked = rerank_diverse(hits, limit=2)
    assert ranked[0].chunk.source.path == "a.py"
    assert ranked[1].chunk.source.path == "b.py"
