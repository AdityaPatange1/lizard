"""Agent primitives for SIMULA.

An :class:`Agent` is a lightweight reasoning participant with a *role*
(drawn from :data:`lizard.constants.AGENT_ROLES`), a *stance* in ``[-1, 1]``
(skeptical .. agreeable) and an *energy* budget in ``[0, 1]``.
An :class:`AgentEnsemble` groups 5-28 agents and exposes bulk accessors.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from lizard.constants import AGENT_ROLES, MAX_AGENTS, MIN_AGENTS

_NAME_PREFIXES: tuple[str, ...] = (
    "Ananda",
    "Buddhi",
    "Chandra",
    "Devi",
    "Eka",
    "Ful",
    "Gita",
    "Hari",
    "Indu",
    "Jaya",
    "Kama",
    "Lila",
    "Maya",
    "Nila",
    "Oja",
    "Priya",
    "Qira",
    "Raga",
    "Sita",
    "Tara",
    "Uma",
    "Vira",
    "Wira",
    "Xana",
    "Yasha",
    "Zana",
    "Atma",
    "Bodhi",
)


def _rotate(seq: tuple[str, ...], offset: int) -> tuple[str, ...]:
    n = len(seq)
    offset %= n
    return seq[offset:] + seq[:offset]


@dataclass
class Agent:
    """A SIMULA agent."""

    name: str
    role: str
    stance: float
    energy: float
    id: int | None = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.stance <= 1.0:
            raise ValueError(f"stance must be in [-1, 1], got {self.stance}")
        if not 0.0 <= self.energy <= 1.0:
            raise ValueError(f"energy must be in [0, 1], got {self.energy}")


@dataclass
class AgentEnsemble:
    """A group of :class:`Agent` values bounded by ``[MIN_AGENTS, MAX_AGENTS]``."""

    agents: list[Agent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        n = len(self.agents)
        if not MIN_AGENTS <= n <= MAX_AGENTS:
            raise ValueError(f"ensemble size must be in [{MIN_AGENTS}, {MAX_AGENTS}], got {n}")

    @classmethod
    def spawn(cls, count: int, rng: random.Random) -> AgentEnsemble:
        """Create a balanced ensemble of ``count`` agents using ``rng``."""
        if not MIN_AGENTS <= count <= MAX_AGENTS:
            raise ValueError(f"count must be in [{MIN_AGENTS}, {MAX_AGENTS}], got {count}")
        offset = rng.randint(0, len(_NAME_PREFIXES) - 1)
        names = _rotate(_NAME_PREFIXES, offset)
        roles = _rotate(AGENT_ROLES, rng.randint(0, len(AGENT_ROLES) - 1))
        agents: list[Agent] = []
        for i in range(count):
            name = f"{names[i % len(names)]}-{i + 1:02d}"
            role = roles[i % len(roles)]
            stance = round(rng.uniform(-1.0, 1.0), 3)
            energy = round(rng.uniform(0.5, 1.0), 3)
            agents.append(Agent(name=name, role=role, stance=stance, energy=energy))
        return cls(agents=agents)

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self) -> Iterator[Agent]:
        return iter(self.agents)

    def stances(self) -> list[float]:
        return [a.stance for a in self.agents]

    def as_rows(self) -> list[tuple[str, str, float, float]]:
        return [(a.name, a.role, a.stance, a.energy) for a in self.agents]

    def by_name(self, name: str) -> Agent:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise KeyError(name)


__all__ = ["Agent", "AgentEnsemble"]
