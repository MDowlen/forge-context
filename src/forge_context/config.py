from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Settings(BaseModel):
    state_dir: Path = Path(".forge-context")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "forge_context"
    embedding_provider: Literal["hash", "sentence-transformers", "openai"] = "hash"
    embedding_model: str | None = None
    embedding_dimensions: int = 256
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    chunk_target_lines: int = 80
    chunk_overlap_lines: int = 12

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("FORGE_EMBEDDING_PROVIDER", "hash").strip().lower()
        if provider not in {"hash", "sentence-transformers", "openai"}:
            raise ValueError(f"Unsupported FORGE_EMBEDDING_PROVIDER: {provider}")

        default_dimensions = 1536 if provider == "openai" else 256
        return cls(
            state_dir=Path(os.getenv("FORGE_CONTEXT_STATE_DIR", ".forge-context")),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "forge_context"),
            embedding_provider=provider,
            embedding_model=os.getenv("FORGE_EMBEDDING_MODEL") or None,
            embedding_dimensions=int(
                os.getenv("FORGE_EMBEDDING_DIMENSIONS", str(default_dimensions))
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            chunk_target_lines=int(os.getenv("FORGE_CHUNK_TARGET_LINES", "80")),
            chunk_overlap_lines=int(os.getenv("FORGE_CHUNK_OVERLAP_LINES", "12")),
        )
