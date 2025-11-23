"""
Enterprise logging configuration with structured output and multiple handlers.

This module provides centralized logging setup for the entire application,
supporting both development (human-readable) and production (JSON) formats.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# Log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that adds context and structure to log records.
    
    In production, this would output JSON for log aggregation systems.
    In development, provides human-readable output with color coding.
    """

    # ANSI color codes for terminal output
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        # Add custom fields if they exist
        provider = getattr(record, "provider", None)
        request_id = getattr(record, "request_id", None)
        user_id = getattr(record, "user_id", None)

        # Build structured message
        parts = [
            f"{record.levelname:8}",
            f"[{record.name}]",
            record.getMessage(),
        ]

        if provider:
            parts.append(f"provider={provider}")
        if request_id:
            parts.append(f"request_id={request_id}")
        if user_id:
            parts.append(f"user_id={user_id}")

        message = " | ".join(parts)

        # Add color in development
        if self.use_color and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            message = f"{color}{message}{self.COLORS['RESET']}"

        return message


def setup_logging(
    level: int = INFO,
    log_file: Optional[Path] = None,
    use_color: bool = True,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Minimum log level to capture (default: INFO)
        log_file: Optional file path for persistent logs
        use_color: Whether to use color in console output (default: True)

    Example:
        >>> setup_logging(level=DEBUG, log_file=Path("logs/validator.log"))
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (stderr for better Unix convention)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(StructuredFormatter(use_color=use_color))
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10_485_760,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        # File logs don't need color
        file_handler.setFormatter(StructuredFormatter(use_color=False))
        root_logger.addHandler(file_handler)

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging initialized",
        extra={
            "level": logging.getLevelName(level),
            "log_file": str(log_file) if log_file else "None",
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Validation started", extra={"provider": "openai"})
    """
    return logging.getLogger(name)
