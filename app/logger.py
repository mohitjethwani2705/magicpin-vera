"""
logger.py — Structured JSON logging setup for the Vera bot.

Every log record is emitted as a single JSON line. The request_id context
variable is automatically injected into every record so that all log lines
within one HTTP request share a common correlation ID.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("context_stored", scope="merchant", context_id="m_001_drmeera")
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Request-scoped correlation ID
# ---------------------------------------------------------------------------

# Set this ContextVar at the top of each request (e.g. in a FastAPI middleware).
# structlog's CallsiteParameter processor will pull it into every log record.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _inject_request_id(
    logger: Any,  # noqa: ANN401  (structlog typing is loose)
    method: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """structlog processor: stamp request_id on every log record."""
    event_dict["request_id"] = request_id_var.get()
    return event_dict


# ---------------------------------------------------------------------------
# One-time configuration (call once at application startup)
# ---------------------------------------------------------------------------

def configure_logging(log_level: str = "INFO") -> None:
    """
    Wire up structlog + stdlib logging so that:
    - structlog-native calls produce JSON
    - Any library that uses stdlib logging (uvicorn, httpx, etc.) also
      produces JSON through the same pipeline.

    Call this exactly once, from main/app startup.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors applied to every log record, regardless of origin.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_request_id,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Hand off to stdlib so the final render uses our stdlib handler.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # stdlib handler that renders as JSON
    formatter = structlog.stdlib.ProcessorFormatter(
        # These processors run only on stdlib-originated records.
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Suppress noisy third-party loggers at WARNING+ to keep output clean.
    for noisy in ("httpx", "httpcore", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structured logger bound to the given name.

    Example:
        log = get_logger(__name__)
        log.info("reply_handled", conversation_id="conv_001", action="send")
        log.warning("auto_reply_detected", count=3, merchant_id="m_001_drmeera")
        log.error("llm_call_failed", exc_info=True)
    """
    return structlog.get_logger(name)
