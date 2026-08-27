# ForgeContext

**Grounded developer context infrastructure for AI engineering workflows.**

ForgeContext turns a source repository into an evidence-backed context layer that developer agents, PR reviewers, CI quality gates, and incident-response systems can reuse. The project is deliberately built as a Python library + CLI first; hosted APIs and dashboards can consume the same core later without coupling the intelligence layer to a web framework.

## Phase 3 capabilities

- Tree-sitter code parsing with symbol-aware chunks
- exact provenance: file, line range, symbol, language, SHA-256
- incremental repository indexing using a content manifest
- pluggable embeddings:
  - deterministic hash embeddings for tests/offline development
  - local `sentence-transformers`
  - OpenAI semantic embeddings
- local persistent vector index or Qdrant
- hybrid semantic + lexical retrieval
- compound-query decomposition and evidence-diversity reranking
- structured grounded answers with citations and confidence
- ADR discovery plus recent Git decision/history context
- local import dependency graph for Python, JavaScript/TypeScript, and Java
- Git-diff-aware reverse impact analysis
- labeled retrieval evaluation with Hit@K and Mean Reciprocal Rank
- evidence-integrity checks for citation file/line pointers
- `ContextEngine` public API for downstream agent workflows
- ForgePR-ready structured `ContextPack`
- GitHub Actions CI across Python 3.11–3.13

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'

dev-ai context sync .
dev-ai context status
dev-ai ask "Where is retrieval scoring implemented?"
dev-ai impact . --changed src/forge_context/retrieval.py
dev-ai context-pack "What changed and what architecture decisions matter?" \
  --changed src/forge_context/retrieval.py
dev-ai eval run evals/example.json
```

### ForgePR-ready Python API

```python
from pathlib import Path

from forge_context import ContextEngine
from forge_context.config import Settings
from forge_context.factory import make_backend, make_embedder

settings = Settings.from_env()
embedder = make_embedder(settings)
backend = make_backend(settings, dimensions=embedder.dimensions)
engine = ContextEngine(backend, embedder)

pack = engine.context_pack(
    Path("."),
    "What behavior changed and what could it affect?",
    changed_files=["src/forge_context/retrieval.py"],
)

print(pack.answer)
print(pack.decisions)
print(pack.impact)
```

### Real local semantic embeddings

```bash
pip install -e '.[dev,local-embeddings]'
export FORGE_EMBEDDING_PROVIDER=sentence-transformers
export FORGE_EMBEDDING_MODEL=all-MiniLM-L6-v2
dev-ai context sync . --force
```

### OpenAI embeddings

Set `OPENAI_API_KEY` securely, then:

```bash
export FORGE_EMBEDDING_PROVIDER=openai
export FORGE_EMBEDDING_MODEL=text-embedding-3-small
export FORGE_EMBEDDING_DIMENSIONS=1536
dev-ai context sync . --force
```

Secrets are never written into the index or manifest.

## Architecture

```text
Repository
   │
   ├── scanner ── excludes generated/vendor content
   ├── Tree-sitter parser ── symbols + code-aware chunks
   ├── manifest ── only changed files are re-embedded
   ├── ADR/Git history ── decision context
   └── dependency graph ── reverse impact context
   │
   ▼
Embedding provider
   ├── hash (CI/offline)
   ├── sentence-transformers (local semantic)
   └── OpenAI (cloud semantic)
   │
   ▼
Vector backend
   ├── local JSON index
   └── Qdrant
   │
   ▼
Query planner + Hybrid Retriever
   ├── vector similarity
   ├── lexical/path/symbol score
   ├── query decomposition
   └── diversity reranking
   │
   ▼
ContextEngine
   ├── GroundedAnswer + exact citations
   ├── decision history
   ├── dependency-aware impact analysis
   └── structured ContextPack for agents
```

## Why the context engine is independent from the agent framework

ForgeContext does not require LangChain, LangGraph, CrewAI, or another orchestration library. Retrieval and provenance are infrastructure contracts. Keeping them independent means the same grounded context can be used by a CLI today, a LangGraph PR-review workflow next, and an incident-triage service later without re-implementing ingestion or evidence tracking.

## Evaluation

`dev-ai eval run` loads labeled questions and reports:

- **Hit@K** — whether an expected source appears in the top K results
- **MRR** — how highly the first correct result was ranked

Phase 3 also adds evidence-integrity verification for citation pointers so downstream automation can check that cited files and line ranges actually exist before acting on a result.

## Development

```bash
pip install -e '.[dev]'
ruff check src tests
pytest -q
```

## Next: ForgePR

ForgeContext v0.3 is the reusable context foundation for the flagship PR validator. The next repository consumes `ContextPack` inside a LangGraph workflow for diff analysis, grounded quality/safety review, test generation, isolated test execution, and a deterministic CI decision gate.

## License

MIT © 2026 Mareza Dowlen
