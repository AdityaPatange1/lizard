"""Tests for agent and ensemble primitives."""

from __future__ import annotations

import random

import pytest

from lizard.constants import MAX_AGENTS, MIN_AGENTS
from lizard.simula.agent import Agent, AgentEnsemble


def test_agent_validates_stance_range() -> None:
    with pytest.raises(ValueError):
        Agent(name="x", role="analyst", stance=1.5, energy=0.5)


def test_agent_validates_energy_range() -> None:
    with pytest.raises(ValueError):
        Agent(name="x", role="analyst", stance=0.0, energy=-0.1)


def test_spawn_ensemble_size_bounds() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError):
        AgentEnsemble.spawn(MIN_AGENTS - 1, rng)
    with pytest.raises(ValueError):
        AgentEnsemble.spawn(MAX_AGENTS + 1, rng)


def test_spawn_ensemble_is_deterministic_under_seed() -> None:
    a = AgentEnsemble.spawn(7, random.Random(42))
    b = AgentEnsemble.spawn(7, random.Random(42))
    assert a.as_rows() == b.as_rows()


def test_by_name_lookup() -> None:
    ensemble = AgentEnsemble.spawn(5, random.Random(1))
    first = next(iter(ensemble))
    assert ensemble.by_name(first.name) is first
    with pytest.raises(KeyError):
        ensemble.by_name("does-not-exist")


def test_stances_length_matches_count() -> None:
    ensemble = AgentEnsemble.spawn(9, random.Random(2))
    assert len(ensemble.stances()) == 9
