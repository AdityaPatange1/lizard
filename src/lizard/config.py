"""Runtime configuration for Lizard.

`LizardConfig` is the single source of truth for paths, limits and model
settings. It is built from environment variables (loaded via python-dotenv)
with sensible defaults so that ``lizard --interactive`` works out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lizard.constants import (
    DEFAULT_AGENT_COUNT,
    DEFAULT_MAX_STEPS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_AGENTS,
    MIN_AGENTS,
)


def _expand(path: str) -> Path:
    """Expand ``~`` and environment variables to an absolute :class:`Path`."""
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_opt_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer or empty, got {raw!r}") from exc


@dataclass(frozen=True)
class LizardConfig:
    """Immutable runtime configuration."""

    home: Path
    db_path: Path
    reports_dir: Path
    configs_dir: Path
    ollama_host: str | None
    ollama_model: str
    agent_count: int
    max_steps: int
    temperature: float
    seed: int | None
    log_level: str

    @classmethod
    def from_env(cls, *, dotenv: bool = True) -> LizardConfig:
        """Load configuration from the process environment.

        Parameters
        ----------
        dotenv:
            If True (default), also load ``.env`` from the current working
            directory before reading variables.
        """
        if dotenv:
            load_dotenv(override=False)

        home = _expand(os.getenv("LIZARD_HOME", "~/.lizard"))
        agent_count = _env_int("LIZARD_AGENT_COUNT", DEFAULT_AGENT_COUNT)
        if not MIN_AGENTS <= agent_count <= MAX_AGENTS:
            raise ValueError(
                f"LIZARD_AGENT_COUNT must be in [{MIN_AGENTS}, {MAX_AGENTS}], got {agent_count}"
            )

        max_steps = _env_int("LIZARD_MAX_STEPS", DEFAULT_MAX_STEPS)
        if max_steps < 1:
            raise ValueError(f"LIZARD_MAX_STEPS must be >= 1, got {max_steps}")

        temperature = _env_float("LIZARD_TEMPERATURE", DEFAULT_TEMPERATURE)
        if temperature <= 0:
            raise ValueError(f"LIZARD_TEMPERATURE must be > 0, got {temperature}")

        ollama_host = os.getenv("OLLAMA_HOST") or None
        ollama_model = (
            os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
        )

        log_level = os.getenv("LIZARD_LOG_LEVEL", "INFO").upper().strip() or "INFO"

        return cls(
            home=home,
            db_path=home / "lizard.db",
            reports_dir=home / "reports",
            configs_dir=home / "configs",
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            agent_count=agent_count,
            max_steps=max_steps,
            temperature=temperature,
            seed=_env_opt_int("LIZARD_SEED"),
            log_level=log_level,
        )

    def ensure_dirs(self) -> None:
        """Create the home, reports and configs directories if missing."""
        for path in (self.home, self.reports_dir, self.configs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def with_agent_count(self, count: int) -> LizardConfig:
        """Return a copy of the config with ``agent_count`` overridden."""
        if not MIN_AGENTS <= count <= MAX_AGENTS:
            raise ValueError(f"agent_count must be in [{MIN_AGENTS}, {MAX_AGENTS}], got {count}")
        return _replace(self, agent_count=count)

    def with_max_steps(self, max_steps: int) -> LizardConfig:
        """Return a copy of the config with ``max_steps`` overridden."""
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        return _replace(self, max_steps=max_steps)


# dataclasses.replace imported lazily to keep the public surface small.
def _replace(cfg: LizardConfig, **changes: Any) -> LizardConfig:  # pragma: no cover - trivial
    from dataclasses import replace

    return replace(cfg, **changes)


__all__ = ["LizardConfig"]
