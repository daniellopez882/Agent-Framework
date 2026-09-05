"""Configuration.

Everything the six step directories hardcoded (``gpt-4o-mini``, ``agent_memory.db``,
``context_db``, "top 3 results") or read ad hoc from ``os.getenv`` is a setting
here, read from the environment or a ``.env`` file.
"""

from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL = "gpt-4o-mini"


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = DEFAULT_MODEL
    OPENAI_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)

    # Web search is optional; the agent runs without it.
    TAVILY_API_KEY: str = ""
    TAVILY_MAX_RESULTS: int = Field(default=3, ge=1, le=10)

    # Upper bound on tool calls in one ``execute``; the loop is real now, so it needs one.
    MAX_TOOL_ITERATIONS: int = Field(default=5, ge=1, le=25)

    AGENT_DB_PATH: str = "agent_memory.db"
    CONTEXT_PERSIST_DIR: str = "context_db"

    @property
    def openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY.strip())

    @property
    def web_search_configured(self) -> bool:
        return bool(self.TAVILY_API_KEY.strip())


def _env_file() -> str | None:
    return None if os.getenv("AGENT_FRAMEWORK_NO_ENV_FILE") else ".env"


settings = Settings(_env_file=_env_file())  # type: ignore[call-arg]


def reload_from_env() -> Settings:
    """Rebuild settings from the current environment (used by the tests)."""
    global settings
    settings = Settings(_env_file=_env_file())  # type: ignore[call-arg]
    return settings
