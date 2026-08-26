from forge_context.models import AnswerBundle, ContextChunk, RetrievalHit, SourceRef


def test_answer_bundle_exposes_evidence():
    source = SourceRef(
        path="src/app.py",
        start_line=2,
        end_line=6,
        symbol="run",
        language="python",
        content_sha256="a" * 64,
    )
    chunk = ContextChunk(id="abc", text="def run(): pass", source=source, kind="code")
    bundle = AnswerBundle(question="where run", hits=[RetrievalHit(chunk=chunk, final_score=0.8)])
    assert bundle.evidence[0].path == "src/app.py"
