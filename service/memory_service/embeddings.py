from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingClient:

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "MEMORY_OPENAI_API_KEY is not set. "
                    "Set it in the environment or .env file to use embeddings."
                )
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._ensure_client()
        try:
            response = await client.embeddings.create(
                input=text,
                model=self._model,
                dimensions=self._dimensions,
            )
            return response.data[0].embedding
        except Exception:
            logger.exception("Embedding failed for text (len=%d)", len(text))
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        try:
            response = await client.embeddings.create(
                input=texts,
                model=self._model,
                dimensions=self._dimensions,
            )
            return [item.embedding for item in response.data]
        except Exception:
            logger.exception("Batch embedding failed for %d texts", len(texts))
            raise
