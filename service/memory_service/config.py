from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 4097
    db_path: str = str(Path.home() / ".opencode" / "memory.db")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    dedupe_threshold: float = 0.92
    decay_factor: float = 0.95
    min_score: float = 0.1
    max_memory_age_days: int = 90

    model_config = {"env_prefix": "MEMORY_", "env_file": ".env"}


settings = Settings()
