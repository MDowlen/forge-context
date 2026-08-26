from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    state_dir: Path = Path(".forge-context")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "forge_context"
    embedding_dimensions: int = 256
    chunk_target_lines: int = 80
    chunk_overlap_lines: int = 12

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            state_dir=Path(os.getenv("FORGE_CONTEXT_STATE_DIR", ".forge-context")),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "forge_context"),
        )
