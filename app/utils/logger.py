"""
AutonomDS Structured Logger
============================
Production-grade logging using structlog with JSON (prod) and 
pretty console (dev) rendering. Every agent gets a bound logger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.logging import RichHandler

from app.utils.config import get_settings, LogFormat

_console = Console(stderr=True)
_initialized = False


def _add_app_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject app-level context into every log record."""
    settings = get_settings()
    event_dict.setdefault("app", settings.app_name)
    event_dict.setdefault("env", settings.app_env.value)
    event_dict.setdefault("version", settings.app_version)
    return event_dict


def setup_logging() -> None:
    """Configure structlog + stdlib logging. Call once at startup."""
    global _initialized
    if _initialized:
        return

    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # ── Shared processors ──────────────────────────────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_app_context,
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == LogFormat.JSON:
        # ── JSON output (production) ───────────────────────────────────
        structlog.configure(
            processors=shared_processors
            + [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            stream=sys.stdout,
        )
    else:
        # ── Rich console output (development) ─────────────────────────
        structlog.configure(
            processors=shared_processors
            + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )

        handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            markup=True,
            show_path=settings.debug,
        )
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

    # ── File handler (optional) ────────────────────────────────────────
    if settings.log_file:
        Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    # ── Suppress noisy third-party loggers ────────────────────────────
    for noisy in ["httpx", "httpcore", "urllib3", "multipart", "matplotlib"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str, **context: Any) -> structlog.BoundLogger:
    """
    Get a named, context-bound logger.

    Usage::

        logger = get_logger("ingestion_agent", dataset="titanic.csv")
        logger.info("loaded dataset", rows=1000, cols=15)
    """
    setup_logging()
    return structlog.get_logger(name).bind(**context)


# ── Module-level convenience ──────────────────────────────────────────────────
setup_logging()
logger = get_logger("autonomds")
