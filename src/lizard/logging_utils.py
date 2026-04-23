"""Structured logging helpers built on top of ``rich``."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_LIZARD_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
        "dhamma.intention": "bright_magenta",
        "dhamma.thought": "bright_cyan",
        "dhamma.speech": "bright_yellow",
        "dhamma.action": "bright_green",
    }
)


def build_console() -> Console:
    """Return a shared, themed :class:`rich.console.Console`."""
    return Console(theme=_LIZARD_THEME, highlight=False, soft_wrap=False)


def configure_logging(level: str = "INFO", *, console: Console | None = None) -> logging.Logger:
    """Configure the ``lizard`` logger.

    Multiple calls are idempotent: handlers are only attached once.
    """
    logger = logging.getLogger("lizard")
    numeric = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logger.setLevel(numeric)

    if not logger.handlers:
        handler = RichHandler(
            console=console or build_console(),
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


__all__ = ["build_console", "configure_logging"]
