from pathlib import Path

from forge_context.backends.local import LocalJsonBackend
from forge_context.config import Settings
from forge_context.embeddings import HashEmbedding
from forge_context.indexer import RepositoryIndexer
from forge_context.retrieval import Retriever


def test_grounded_answer_contains_source_pointer(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "fallback.py").write_text("def fallback():\n    return 'safe'\n")

    settings = Settings(state_dir=tmp_path / "state", embedding_dimensions=64)
    backend = LocalJsonBackend(settings.state_dir)
    embedder = HashEmbedding(64)
    RepositoryIndexer(backend, settings, embedder).sync(repo)

    answer = Retriever(backend, embedder).grounded_answer("Where is fallback handled?", limit=1)
    assert answer.citations[0].path == "fallback.py"
    assert "fallback.py" in answer.answer
