from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .backends.base import VectorBackend
from .config import Settings
from .embeddings import EmbeddingProvider
from .models import IndexedChunk, SyncReport
from .parser import chunk_file
from .scanner import discover_files


class RepositoryIndexer:
    def __init__(self, backend: VectorBackend, settings: Settings, embedder: EmbeddingProvider) -> None:
        self.backend = backend
        self.settings = settings
        self.embedder = embedder
        self.manifest_path = settings.state_dir / "manifest.json"

    @staticmethod
    def _file_sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load_manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return {str(path): str(value) for path, value in raw.items()}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def sync(self, root: Path, force: bool = False) -> SyncReport:
        root = root.resolve()
        scan = discover_files(root)
        previous_manifest = {} if force else self._load_manifest()
        current_manifest: dict[str, str] = {}
        path_lookup: dict[str, Path] = {}

        for file in scan.files:
            rel = file.resolve().relative_to(root).as_posix()
            path_lookup[rel] = file
            current_manifest[rel] = self._file_sha(file)

        previous_paths = set(previous_manifest)
        current_paths = set(current_manifest)
        deleted = previous_paths - current_paths
        added = current_paths - previous_paths
        changed = {
            path
            for path in current_paths & previous_paths
            if current_manifest[path] != previous_manifest[path]
        }
        unchanged = current_paths - added - changed

        existing = [] if force else self.backend.all()
        kept = [chunk for chunk in existing if chunk.source.path in unchanged]
        new_chunks: list[IndexedChunk] = []
        languages: Counter[str] = Counter()
        files_indexed = 0

        for chunk in kept:
            languages[chunk.source.language or chunk.kind] += 1

        for rel in sorted(added | changed):
            file = path_lookup[rel]
            try:
                chunks = chunk_file(
                    file,
                    root,
                    target_lines=self.settings.chunk_target_lines,
                    overlap_lines=self.settings.chunk_overlap_lines,
                )
            except (OSError, UnicodeError):
                continue
            if not chunks:
                continue
            files_indexed += 1
            texts = [chunk.text for chunk in chunks]
            vectors = self.embedder.embed_many(texts)
            for chunk, vector in zip(chunks, vectors, strict=True):
                language = chunk.source.language or chunk.kind
                languages[language] += 1
                new_chunks.append(IndexedChunk(**chunk.model_dump(), vector=vector))

        merged = kept + new_chunks
        self.backend.replace(merged)
        self._save_manifest(current_manifest)
        return SyncReport(
            root=str(root),
            files_seen=len(scan.files),
            files_indexed=files_indexed,
            chunks_indexed=len(merged),
            skipped_files=scan.skipped,
            languages=dict(languages),
            added_files=len(added),
            changed_files=len(changed),
            unchanged_files=len(unchanged),
            deleted_files=len(deleted),
        )
