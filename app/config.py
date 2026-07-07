"""
config.py — Central settings for the Vera bot.

All configuration is read from environment variables (with sensible defaults).
Import the `settings` singleton; never instantiate Settings directly.

Usage:
    from config import settings
    # settings.groq_api_key and settings.groq_model are used by conversation_handlers.py
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Load .env file manually (no external dependency)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())


class Settings:
    """
    Flat settings bag loaded from environment variables.

    We use plain os.environ rather than pydantic-settings so that the module
    has zero heavy dependencies at import time — it must load before the app
    (and thus before pydantic is guaranteed to have been imported).
    """

    def __init__(self) -> None:
        # --- Groq inference ---
        self.groq_api_key: str = os.environ["GROQ_API_KEY"]  # required — fail fast at startup
        self.groq_model: str = os.environ.get(
            "GROQ_MODEL", "llama-3.1-8b-instant"
        )

        # --- Server ---
        self.host: str = os.environ.get("HOST", "0.0.0.0")
        self.port: int = int(os.environ.get("PORT", "8000"))
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper()

        # --- Bot identity ---
        self.team_name: str = os.environ.get("TEAM_NAME", "YourTeamName")
        self.contact_email: str = os.environ.get("CONTACT_EMAIL", "team@example.com")
        self.bot_version: str = os.environ.get("BOT_VERSION", "1.0.0")

        # --- Conversation tuning ---
        # How many identical (or near-identical) messages from the same party
        # before we declare it an auto-reply loop and exit gracefully.
        self.auto_reply_threshold: int = int(
            os.environ.get("AUTO_REPLY_THRESHOLD", "3")
        )
        # Maximum turns before we force a graceful close regardless of state.
        self.max_conversation_turns: int = int(
            os.environ.get("MAX_CONVERSATION_TURNS", "10")
        )
        # Similarity ratio (0-1) above which two messages are considered "same".
        self.auto_reply_similarity_threshold: float = float(
            os.environ.get("AUTO_REPLY_SIMILARITY_THRESHOLD", "0.85")
        )

        # --- LLM generation ---
        self.llm_max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "512"))
        self.llm_temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.3"))


@lru_cache(maxsize=1)
def _build_settings() -> Settings:
    return Settings()


# Module-level singleton — import this everywhere.
settings: Settings = _build_settings()
