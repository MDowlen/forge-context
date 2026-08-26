# ForgeContext

ForgeContext is the shared developer-context layer for a larger AI engineering portfolio suite. It indexes a software repository, extracts code structure with Tree-sitter, stores grounded chunks with provenance, and exposes retrieval through a CLI.

The long-term suite will reuse this engine for:

- a RAG-powered PR and CI/CD validator,
- an intelligent developer CLI,
- and an incident-triage/root-cause analysis system.

## Phase 1 status

Implemented in this foundation:

- repository discovery with sensible ignore rules,
- Tree-sitter symbol extraction for supported languages,
- provenance-rich code/document chunks,
- deterministic local embedding fallback for tests and zero-network development,
- Qdrant-ready vector-store abstraction,
- lexical + vector hybrid ranking,
- CLI commands for indexing and querying,
- typed Pydantic contracts for every retrieval result,
- automated tests for scanning, indexing, retrieval, and provenance.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest

dev-ai context sync .
dev-ai ask "Where is repository scanning implemented?"
```

## Commands

```bash
dev-ai context sync PATH
dev-ai context status
dev-ai ask "question"
dev-ai explain FILE[:LINE]
```

## Design principle

Every answer is grounded in retrievable evidence. Retrieval results carry file paths, line ranges, content hashes, symbol names when available, and a score. Downstream agents should never receive unsupported free-form context without provenance.
