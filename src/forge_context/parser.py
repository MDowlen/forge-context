from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .models import ContextChunk, SourceRef, relative_path


LANG_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
}

DOC_SUFFIXES = {".md", ".rst", ".txt"}
CONFIG_SUFFIXES = {".toml", ".yaml", ".yml", ".json"}


@dataclass(frozen=True)
class SymbolSpan:
    name: str
    start_line: int
    end_line: int


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _chunk_id(path: str, start: int, end: int, digest: str) -> str:
    raw = f"{path}:{start}:{end}:{digest}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _fallback_symbols(text: str, language: str | None) -> list[SymbolSpan]:
    if language == "python":
        pattern = re.compile(r"^(?:async\s+)?(?:def|class)\s+([A-Za-z_][\w]*)", re.MULTILINE)
    elif language in {"javascript", "typescript", "tsx"}:
        pattern = re.compile(
            r"^(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)",
            re.MULTILINE,
        )
    elif language == "java":
        pattern = re.compile(r"^\s*(?:public|protected|private)?\s*(?:class|interface|enum)\s+([A-Za-z_]\w*)", re.MULTILINE)
    else:
        return []

    starts = [(m.group(1), text[: m.start()].count("\n") + 1) for m in pattern.finditer(text)]
    lines = text.splitlines()
    spans: list[SymbolSpan] = []
    for idx, (name, start) in enumerate(starts):
        next_start = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines) + 1
        spans.append(SymbolSpan(name=name, start_line=start, end_line=max(start, next_start - 1)))
    return spans


def extract_symbols(text: str, language: str | None) -> list[SymbolSpan]:
    if not language:
        return []
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(language)
        tree = parser.parse(text.encode("utf-8", errors="replace"))
        query_types = {
            "python": {"function_definition", "class_definition"},
            "javascript": {"function_declaration", "class_declaration", "method_definition"},
            "typescript": {"function_declaration", "class_declaration", "method_definition"},
            "tsx": {"function_declaration", "class_declaration", "method_definition"},
            "java": {"class_declaration", "interface_declaration", "method_declaration"},
            "go": {"function_declaration", "method_declaration", "type_declaration"},
            "rust": {"function_item", "struct_item", "enum_item", "impl_item"},
        }.get(language, set())
        if not query_types:
            return _fallback_symbols(text, language)

        spans: list[SymbolSpan] = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            stack.extend(reversed(node.children))
            if node.type not in query_types:
                continue
            name_node = node.child_by_field_name("name")
            name = text[name_node.start_byte : name_node.end_byte] if name_node else node.type
            spans.append(
                SymbolSpan(
                    name=name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            )
        return sorted(spans, key=lambda x: (x.start_line, x.end_line)) or _fallback_symbols(text, language)
    except Exception:
        return _fallback_symbols(text, language)


def _kind_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in LANG_BY_SUFFIX:
        return "code"
    if suffix in DOC_SUFFIXES:
        return "document"
    if suffix in CONFIG_SUFFIXES:
        return "config"
    return "unknown"


def chunk_file(path: Path, root: Path, target_lines: int = 80, overlap_lines: int = 12) -> list[ContextChunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    rel = relative_path(path, root)
    language = LANG_BY_SUFFIX.get(path.suffix.lower())
    symbols = extract_symbols(text, language)
    kind = _kind_for(path)

    chunks: list[ContextChunk] = []
    step = max(1, target_lines - overlap_lines)
    for start_idx in range(0, max(len(lines), 1), step):
        end_idx = min(len(lines), start_idx + target_lines)
        if end_idx <= start_idx:
            break
        chunk_text = "\n".join(lines[start_idx:end_idx]).strip()
        if not chunk_text:
            continue
        start_line = start_idx + 1
        end_line = end_idx
        symbol = next(
            (
                item.name
                for item in symbols
                if item.start_line <= start_line <= item.end_line
                or start_line <= item.start_line <= end_line
            ),
            None,
        )
        digest = _sha(chunk_text)
        chunks.append(
            ContextChunk(
                id=_chunk_id(rel, start_line, end_line, digest),
                text=chunk_text,
                source=SourceRef(
                    path=rel,
                    start_line=start_line,
                    end_line=end_line,
                    symbol=symbol,
                    language=language,
                    content_sha256=digest,
                ),
                kind=kind,
                metadata={"suffix": path.suffix.lower()},
            )
        )
        if end_idx == len(lines):
            break

    return chunks
