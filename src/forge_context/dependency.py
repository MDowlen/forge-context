from __future__ import annotations

import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path

from .models import DependencyEdge, ImpactedFile, ImpactReport
from .parser import LANG_BY_SUFFIX
from .scanner import discover_files


PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))", re.MULTILINE)
JS_IMPORT_RE = re.compile(
    r"(?:from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\)|import\(\s*['\"]([^'\"]+)['\"]\s*\))"
)
JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_\.]+);", re.MULTILINE)


def _module_candidates(source: str, import_name: str, language: str | None) -> list[str]:
    parent = Path(source).parent
    candidates: list[str] = []
    if language == "python":
        normalized = import_name.replace(".", "/")
        candidates.extend([f"{normalized}.py", f"{normalized}/__init__.py"])
        candidates.extend(
            [
                (parent / f"{normalized}.py").as_posix(),
                (parent / normalized / "__init__.py").as_posix(),
            ]
        )
    elif language in {"javascript", "typescript", "tsx"}:
        if not import_name.startswith("."):
            return []
        stem = (parent / import_name).as_posix()
        for suffix in [".js", ".jsx", ".ts", ".tsx"]:
            candidates.append(f"{stem}{suffix}")
        for suffix in [".js", ".jsx", ".ts", ".tsx"]:
            candidates.append(f"{stem}/index{suffix}")
    elif language == "java":
        normalized = import_name.replace(".", "/")
        candidates.append(f"{normalized}.java")
    return [str(Path(item)) for item in candidates]


def _imports(text: str, language: str | None) -> list[str]:
    if language == "python":
        return [a or b for a, b in PY_IMPORT_RE.findall(text)]
    if language in {"javascript", "typescript", "tsx"}:
        return [a or b or c for a, b, c in JS_IMPORT_RE.findall(text)]
    if language == "java":
        return JAVA_IMPORT_RE.findall(text)
    return []


def build_dependency_edges(root: Path) -> list[DependencyEdge]:
    root = root.resolve()
    scan = discover_files(root)
    local_paths = {path.resolve().relative_to(root).as_posix() for path in scan.files}
    edges: list[DependencyEdge] = []
    seen: set[tuple[str, str, str]] = set()

    for path in scan.files:
        language = LANG_BY_SUFFIX.get(path.suffix.lower())
        if language not in {"python", "javascript", "typescript", "tsx", "java"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        source = path.resolve().relative_to(root).as_posix()
        for import_name in _imports(text, language):
            target = next(
                (candidate for candidate in _module_candidates(source, import_name, language) if candidate in local_paths),
                None,
            )
            if target is None:
                continue
            key = (source, target, import_name)
            if key in seen:
                continue
            seen.add(key)
            edges.append(DependencyEdge(source=source, target=target, import_name=import_name))
    return edges


def git_changed_files(root: Path, base: str = "HEAD~1", head: str = "HEAD") -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def analyze_impact(
    root: Path,
    changed_files: list[str],
    max_depth: int = 4,
) -> ImpactReport:
    edges = build_dependency_edges(root)
    reverse: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        reverse[edge.target].add(edge.source)

    queue: deque[tuple[str, int]] = deque((path, 0) for path in changed_files)
    best_depth: dict[str, int] = {}
    reasons: dict[str, str] = {}
    while queue:
        current, depth = queue.popleft()
        previous = best_depth.get(current)
        if previous is not None and previous <= depth:
            continue
        best_depth[current] = depth
        reasons[current] = "changed directly" if depth == 0 else reasons.get(current, "depends on changed code")
        if depth >= max_depth:
            continue
        for dependent in sorted(reverse.get(current, set())):
            next_depth = depth + 1
            if dependent not in best_depth or next_depth < best_depth[dependent]:
                reasons[dependent] = f"depends on {current}"
                queue.append((dependent, next_depth))

    impacted = [
        ImpactedFile(path=path, depth=depth, reason=reasons[path])
        for path, depth in sorted(best_depth.items(), key=lambda item: (item[1], item[0]))
    ]
    return ImpactReport(
        changed_files=sorted(changed_files),
        impacted_files=impacted,
        edges_considered=len(edges),
    )
