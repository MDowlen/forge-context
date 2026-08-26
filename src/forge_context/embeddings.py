from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from typing import Protocol


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedding:
    """Deterministic zero-network embedding for local development and tests.

    This provider is intentionally simple. It keeps ForgeContext reproducible in
    CI and offline environments while preserving the same interface used by real
    semantic embedding providers.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            idx = value % self.dimensions
            sign = 1.0 if (value >> 1) & 1 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class SentenceTransformerEmbedding:
    """Local semantic embeddings backed by sentence-transformers.

    The dependency stays optional so the core CLI remains lightweight. Install
    ForgeContext with ``pip install -e '.[local-embeddings]'`` to enable it.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "sentence-transformers is not installed; use `pip install -e '.[local-embeddings]'`"
            ) from exc
        self.model_name = model
        self._model = SentenceTransformer(model)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Could not determine embedding dimension for {model}")
        self.dimensions = int(dimension)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


class OpenAIEmbedding:
    """Semantic embeddings using the OpenAI embeddings HTTP API.

    No SDK dependency is required. The API key is read by configuration and is
    never persisted in the ForgeContext index or manifest.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network integration
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI embeddings request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network integration
            raise RuntimeError(f"OpenAI embeddings request failed: {exc.reason}") from exc

        data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
        if len(data) != len(texts):
            raise RuntimeError("OpenAI embeddings response count did not match input count")
        return [list(map(float, item["embedding"])) for item in data]
