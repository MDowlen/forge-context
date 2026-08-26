from pathlib import Path

from forge_context.backends.local import LocalJsonBackend
from forge_context.config import Settings
from forge_context.embeddings import HashEmbedding
from forge_context.indexer import RepositoryIndexer
from forge_context.retrieval import Retriever


def test_index_and_retrieve_exact_concept(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "payments.py").write_text(
        "def retry_payment():\n    # exponential backoff for payment gateway\n    return 'retry'\n"
    )
    (repo / "users.py").write_text("def create_user():\n    return 'user'\n")

    settings = Settings(state_dir=tmp_path / "state", embedding_dimensions=64)
    backend = LocalJsonBackend(settings.state_dir)
    embedder = HashEmbedding(settings.embedding_dimensions)
    report = RepositoryIndexer(backend, settings, embedder).sync(repo)

    assert report.files_indexed == 2
    assert backend.count() >= 2

    result = Retriever(backend, embedder).ask("Where is payment retry backoff implemented?", limit=1)
    assert result.hits[0].chunk.source.path == "payments.py"
    assert result.hits[0].chunk.source.start_line == 1
