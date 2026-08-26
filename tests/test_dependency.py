from pathlib import Path

from forge_context.dependency import analyze_impact, build_dependency_edges


def test_dependency_graph_and_reverse_impact(tmp_path: Path):
    (tmp_path / "core.py").write_text("def run():\n    return 1\n")
    (tmp_path / "service.py").write_text("from core import run\n\ndef service():\n    return run()\n")
    (tmp_path / "api.py").write_text("import service\n\ndef handler():\n    return service.service()\n")

    edges = build_dependency_edges(tmp_path)
    pairs = {(edge.source, edge.target) for edge in edges}
    assert ("service.py", "core.py") in pairs
    assert ("api.py", "service.py") in pairs

    report = analyze_impact(tmp_path, ["core.py"])
    depths = {item.path: item.depth for item in report.impacted_files}
    assert depths["core.py"] == 0
    assert depths["service.py"] == 1
    assert depths["api.py"] == 2
