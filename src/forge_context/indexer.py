from __future__ import annotations

from collections import Counter
from pathlib import Path

from .backends.base import VectorBackend
from .config import Settings
from .embeddings import HashEmbedding
from .models import IndexedChunk, SyncReport
from .parser import chunk_file
from .scanner import discover_files


class RepositoryIndexer:
    def __init__(self, backend: VectorBackend, settings: Settings, embedder: HashEmbedding | None = None) -> None:
        self.backend = backend
        self.settings = settings
        self.embedder = embedder or HashEmbedding(settings.embedding_dimensions)

    def sync(self, root: Path) -> SyncReport:
        root = root.resolve()
        scan = discover_files(root)
        indexed: list[IndexedChunk] = []
        languages: Counter[str] = Counter()
        files_indexed = 0

        for file in scan.files:
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
            for chunk in chunks:
                language = chunk.source.language or chunk.kind
                languages[language] += 1
                indexed.append(
                    IndexedChunk(
                        **chunk.model_dump(),
                        vector=self.embedder.embed(chunk.text),
                    )
                )

        self.backend.replace(indexed)
        return SyncReport(
            root=str(root),
            files_seen=len(scan.files),
            files_indexed=files_indexed,
            chunks_indexed=len(indexed),
            skipped_files=scan.skipped,
            languages=dict(languages),
        )
