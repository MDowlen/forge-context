from pathlib import Path

from forge_context.parser import chunk_file


def test_chunk_preserves_provenance_and_symbol(tmp_path: Path):
    path = tmp_path / "service.py"
    path.write_text("def retry_request():\n    return 3\n\nclass Client:\n    pass\n")
    chunks = chunk_file(path, tmp_path, target_lines=20, overlap_lines=0)
    assert len(chunks) == 1
    assert chunks[0].source.path == "service.py"
    assert chunks[0].source.start_line == 1
    assert chunks[0].source.end_line == 5
    assert chunks[0].source.symbol in {"retry_request", "Client"}
    assert len(chunks[0].source.content_sha256) == 64
