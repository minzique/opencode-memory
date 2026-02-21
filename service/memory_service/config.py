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

    extraction_model: str = "gemini-2.5-flash-lite"
    extraction_api_key: str = ""  # falls back to openai_api_key if empty
    extraction_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    consolidation_threshold: float = 0.85

    dedupe_threshold: float = 0.92
    decay_factor: float = 0.95
    min_score: float = 0.1
    max_memory_age_days: int = 90

    # Background decay scheduler
    decay_interval_hours: float = 24.0  # run decay every N hours (0 = disabled)

    # Cross-project sharing guardrails
    cross_project_enabled: bool = True
    cross_project_types: list[str] = [
        "preference", "convention", "pattern", "architecture", "fact",
    ]
    cross_project_blocked_types: list[str] = [
        "constraint", "decision", "error-solution", "working_context",
    ]
    cross_project_threshold: float = 0.5  # higher bar than same-project (0.3)
    cross_project_budget_pct: float = 0.2  # max 20% of memory slots

    blog_repo_path: str = str(Path.home() / "Developer" / "log")
    blog_content_dir: str = "src/content/blog"
    blog_default_author: str = "Lume"

    model_config = {"env_prefix": "MEMORY_", "env_file": ".env"}


settings = Settings()
