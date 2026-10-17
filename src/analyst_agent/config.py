"""Settings. Every value can be set with an AGENT_* environment variable or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "anthropic", "bedrock"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    # LLM provider. "fake" is deterministic and free: it answers golden questions with their gold SQL,
    # so it exercises the whole pipeline but says nothing about model quality.
    provider: Provider = "fake"
    model: str = "claude-haiku-4-5-20251001"
    bedrock_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    aws_region: str = "us-east-1"
    max_output_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # Warehouse (read-only). DuckDB by default; the demo file is built by scripts/build_demo_warehouse.py.
    warehouse_path: Path = Path("data/demo_warehouse.duckdb")
    allowed_schemas: list[str] = Field(default_factory=lambda: ["main_marts", "marts"])
    row_limit: int = 200
    query_timeout_seconds: float = 20.0
    # Human-in-the-loop: a query whose estimated scan exceeds this many rows and has no date filter
    # pauses the graph until someone approves it.
    approval_row_threshold: int = 100_000

    # API
    api_keys: list[str] = Field(
        default_factory=list, description="Accepted X-API-Key values; empty = open (dev)"
    )
    rate_limit: str = "30/minute"
    cors_origins: list[str] = Field(default_factory=list)

    # Memory. Postgres when set; otherwise an in-process saver (tests, laptops).
    database_url: str | None = None

    # Pricing in USD per million tokens, used for the per-request cost figure.
    price_input_per_mtok: float = 1.00
    price_output_per_mtok: float = 5.00

    service_name: str = "analyst-agent"
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
