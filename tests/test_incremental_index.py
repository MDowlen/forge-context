from pathlib import Path

from forge_context.backends.local import LocalJsonBackend
from forge_context.config import Settings
from forge_context.embeddings import HashEmbedding
from forge_context.indexer import RepositoryIndexer


def test_incremental_sync_reuses_unchanged_files(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def alpha():\n    return 1\n")
    (repo / "b.py").write_text("def beta():\n    return 2\n")

    settings = Settings(state_dir=tmp_path / "state", embedding_dimensions=64)
    backend = LocalJsonBackend(settings.state_dir)
    embedder = HashEmbedding(64)
    indexer = RepositoryIndexer(backend, settings, embedder)

    first = indexer.sync(repo)
    assert first.added_files == 2
    assert first.unchanged_files == 0

    second = indexer.sync(repo)
    assert second.added_files == 0
    assert second.changed_files == 0
    assert second.unchanged_files == 2
    assert second.files_indexed == 0

    (repo / "a.py").write_text("def alpha():\n    return 3\n")
    third = indexer.sync(repo)
    assert third.changed_files == 1
    assert third.unchanged_files == 1
    assert third.files_indexed == 1
