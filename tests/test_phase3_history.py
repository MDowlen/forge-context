from pathlib import Path

from forge_context.history import discover_decision_docs


def test_discovers_architecture_decision_records(tmp_path: Path):
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    (adr_dir / "ADR-001-vector-store.md").write_text(
        "# Use Qdrant for remote vector search\n\nWe need durable semantic retrieval.\n",
        encoding="utf-8",
    )

    records = discover_decision_docs(tmp_path)

    assert len(records) == 1
    assert records[0].source == "adr"
    assert records[0].reference == "docs/adr/ADR-001-vector-store.md"
    assert "Qdrant" in records[0].title
