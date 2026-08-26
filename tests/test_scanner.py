from pathlib import Path

from forge_context.scanner import discover_files


def test_scanner_ignores_generated_directories(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("junk")
    result = discover_files(tmp_path)
    assert [p.name for p in result.files] == ["app.py"]
