from pathlib import Path

from forge_context.backends.local import LocalJsonBackend
from forge_context.config import Settings
from forge_context.embeddings import HashEmbedding
from forge_context.evaluation import evaluate_retrieval
from forge_context.indexer import RepositoryIndexer
from forge_context.models import EvalCase
from forge_context.retrieval import Retriever


def test_retrieval_eval_reports_hit_rate_and_mrr(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "retry.py").write_text("def backoff():\n    # retry exponential payment backoff\n    return 2\n")
    (repo / "user.py").write_text("def user():\n    return 1\n")

    settings = Settings(state_dir=tmp_path / "state", embedding_dimensions=64)
    backend = LocalJsonBackend(settings.state_dir)
    embedder = HashEmbedding(64)
    RepositoryIndexer(backend, settings, embedder).sync(repo)

    cases = [EvalCase(question="payment retry backoff", expected_paths=["retry.py"])]
    report = evaluate_retrieval(Retriever(backend, embedder), cases, k=2)
    assert report.hit_rate_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
