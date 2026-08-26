from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".forge-context",
}

SUPPORTED_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}


@dataclass(frozen=True)
class ScanResult:
    files: list[Path]
    skipped: int


def discover_files(root: Path) -> ScanResult:
    root = root.resolve()
    files: list[Path] = []
    skipped = 0

    for path in root.rglob("*"):
        if any(part in DEFAULT_IGNORES for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped += 1
            continue
        try:
            if path.stat().st_size > 2_000_000:
                skipped += 1
                continue
        except OSError:
            skipped += 1
            continue
        files.append(path)

    return ScanResult(files=sorted(files), skipped=skipped)
