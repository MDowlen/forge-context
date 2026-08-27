from pathlib import Path

from forge_context.evaluation import evaluate_evidence_integrity
from forge_context.models import Citation, GroundedAnswer


def test_evidence_integrity_validates_real_source_pointers(tmp_path: Path):
    source = tmp_path / "app.py"
    source.write_text("line1\nline2\nline3\n", encoding="utf-8")
    answer = GroundedAnswer(
        question="Where?",
        answer="Grounded in app.py",
        citations=[Citation(path="app.py", start_line=1, end_line=2, score=0.9)],
        confidence=0.9,
    )

    report = evaluate_evidence_integrity(tmp_path, answer)

    assert report.citations == 1
    assert report.valid_pointers == 1
    assert report.evidence_integrity == 1.0


def test_evidence_integrity_rejects_out_of_range_pointer(tmp_path: Path):
    (tmp_path / "app.py").write_text("line1\n", encoding="utf-8")
    answer = GroundedAnswer(
        question="Where?",
        answer="Bad pointer",
        citations=[Citation(path="app.py", start_line=1, end_line=9, score=0.5)],
        confidence=0.5,
    )

    assert evaluate_evidence_integrity(tmp_path, answer).evidence_integrity == 0.0
