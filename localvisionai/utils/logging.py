"""Structured logging setup for LocalVisionAI.

Usage:
    from localvisionai.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("pipeline started", extra={"job_id": "j_abc123"})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Emits one JSON object per log record — useful for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields attached via logger.info(..., extra={...})
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                payload[key] = val
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable colored formatter for terminal output (uses rich-style)."""

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        prefix = f"{color}[{ts}] [{record.levelname}]{self.RESET} {record.name}:"
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        return f"{prefix} {message}"


def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure root logging for the package.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_logs: If True, emit JSON lines. Otherwise, use human-readable format.
        log_file: Optional file path to also write logs to.
    """
    root_logger = logging.getLogger("localvisionai")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(
        JSONFormatter() if json_logs else HumanFormatter()
    )
    root_logger.addHandler(console_handler)

    # Optional file handler (always JSON for structured parsing)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the localvisionai namespace."""
    if not name.startswith("localvisionai"):
        name = f"localvisionai.{name}"
    return logging.getLogger(name)
