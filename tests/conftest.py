"""Shared pytest fixtures."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from lizard.config import LizardConfig
from lizard.llm import LLMClient
from lizard.simula.engine import Simula
from lizard.storage.db import Database
from lizard.storage.files import ReportFiles


@pytest.fixture
def lizard_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LIZARD_HOME", str(tmp_path))
    monkeypatch.setenv("LIZARD_SEED", "42")
    monkeypatch.setenv("LIZARD_AGENT_COUNT", "5")
    monkeypatch.setenv("LIZARD_MAX_STEPS", "2")
    monkeypatch.setenv("LIZARD_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("OLLAMA_HOST", "")
    monkeypatch.setenv("LIZARD_DISABLE_OLLAMA", "1")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    os.environ.setdefault("OLLAMA_MODEL", "gemma3")
    return tmp_path


@pytest.fixture
def config(lizard_home: Path) -> LizardConfig:
    cfg = LizardConfig.from_env(dotenv=False)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def database(config: LizardConfig) -> Database:
    return Database(config.db_path)


@pytest.fixture
def files(config: LizardConfig) -> ReportFiles:
    return ReportFiles(reports_dir=config.reports_dir)


@pytest.fixture
def offline_llm() -> LLMClient:
    rng = random.Random(0)
    client = LLMClient(model="gemma3", rng=rng)
    # Force the deterministic fallback path regardless of ollama availability.
    client._client = None  # type: ignore[attr-defined]
    return client


@pytest.fixture
def simula(config: LizardConfig, database: Database, offline_llm: LLMClient) -> Iterator[Simula]:
    engine = Simula(
        config=config,
        db=database,
        llm=offline_llm,
        rng=random.Random(config.seed),
    )
    yield engine
