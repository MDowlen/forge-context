from __future__ import annotations

import subprocess
from pathlib import Path

from .models import DecisionRecord

ADR_HINTS = {"adr", "adrs", "decision", "decisions", "architecture-decisions"}


def collect_git_history(root: Path, limit: int = 20) -> list[DecisionRecord]:
    root = root.resolve()
    result = subprocess.run(
        [
            "git",
            "log",
            f"-{limit}",
            "--date=iso-strict",
            "--pretty=format:%H%x1f%ad%x1f%s",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    records: list[DecisionRecord] = []
    for line in result.stdout.splitlines():
        parts = line.split("\x1f", 2)
        if len(parts) != 3:
            continue
        sha, timestamp, title = parts
        records.append(
            DecisionRecord(
                source="git",
                title=title.strip(),
                reference=sha.strip(),
                timestamp=timestamp.strip() or None,
            )
        )
    return records


def discover_decision_docs(root: Path, limit: int = 20) -> list[DecisionRecord]:
    root = root.resolve()
    records: list[DecisionRecord] = []
    for path in root.rglob("*.md"):
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        lowered_parts = {part.lower() for part in relative.parts}
        name = path.stem.lower()
        if not (lowered_parts & ADR_HINTS or name.startswith("adr-") or name.startswith("adr_")):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        nonempty = [line.strip() for line in text.splitlines() if line.strip()]
        title = nonempty[0].lstrip("# ") if nonempty else path.stem
        summary = " ".join(line.lstrip("# ") for line in nonempty[1:4])[:500]
        records.append(
            DecisionRecord(
                source="adr",
                title=title,
                reference=relative.as_posix(),
                summary=summary,
            )
        )
        if len(records) >= limit:
            break
    return records


def collect_decision_context(root: Path, limit: int = 20) -> list[DecisionRecord]:
    adrs = discover_decision_docs(root, limit=limit)
    remaining = max(0, limit - len(adrs))
    return adrs + collect_git_history(root, limit=remaining)
