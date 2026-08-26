# ForgeContext architecture

## Pipeline

1. **Scanner** discovers source, docs, and config files while excluding generated/vendor folders.
2. **Parser** uses Tree-sitter when available to identify symbols and language structure.
3. **Chunker** creates bounded line-range chunks while preserving file, symbol, language, line range, and SHA-256 provenance.
4. **Embedder** converts chunks into vectors. Phase 1 ships with a deterministic zero-network fallback so development and tests are reproducible.
5. **Vector backend** persists the index locally or in Qdrant through a shared interface.
6. **Retriever** performs vector candidate retrieval and combines it with lexical/path/symbol scoring.
7. **CLI** exposes sync, status, ask, and explain workflows.

## Why this matters for the later agents

The PR reviewer, test generator, and incident-triage workflows should consume `RetrievalHit` objects rather than raw model-generated context. This creates an evidence boundary: every piece of context has a source path and line range that can be cited, evaluated, and audited.

## Next milestone

- semantic local embeddings,
- incremental indexing keyed by file hashes,
- git-diff impact analysis,
- dependency graph edges from Tree-sitter symbols/imports,
- structured answer generation with citations,
- retrieval evaluation dataset and metrics.
