"""A small catalogue of world technologies that flavour agent reasoning.

Each :class:`Technology` declares which Dhamma phases it is appropriate for
(from :data:`lizard.constants.DHAMMA_PHASES`). The engine picks one per
phase via the agent's RNG so runs are deterministic under a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Technology:
    """A single affordance used to bias agent reasoning."""

    name: str
    description: str
    phases: tuple[str, ...]


CATALOGUE: tuple[Technology, ...] = (
    Technology(
        name="jurisprudence",
        description="Frames the situation in terms of statute, precedent and due process.",
        phases=("right_intention", "right_thought", "right_speech", "right_action"),
    ),
    Technology(
        name="forensics",
        description="Evidence-first investigation, chain of custody, corroboration.",
        phases=("right_thought", "right_action"),
    ),
    Technology(
        name="ethics",
        description="Virtue-, deontological-, and consequentialist lenses on intent.",
        phases=("right_intention", "right_thought"),
    ),
    Technology(
        name="rhetoric",
        description="Clear, charitable, non-inflammatory communication.",
        phases=("right_speech",),
    ),
    Technology(
        name="game_theory",
        description="Analyses incentives, cooperation, and equilibria between parties.",
        phases=("right_thought", "right_action"),
    ),
    Technology(
        name="thermodynamics",
        description="Weighs entropy, uncertainty and free energy of the decision.",
        phases=("right_thought",),
    ),
    Technology(
        name="search",
        description="Enumerates plans via BFS/DFS/A* over the consequence tree.",
        phases=("right_thought", "right_action"),
    ),
    Technology(
        name="psychology",
        description="Models motivations, biases, and emotional regulation.",
        phases=("right_intention", "right_speech"),
    ),
    Technology(
        name="systems_theory",
        description="Treats the scenario as a feedback system with stakeholders.",
        phases=("right_thought", "right_action"),
    ),
    Technology(
        name="mediation",
        description="Designs a procedurally fair communication protocol.",
        phases=("right_speech", "right_action"),
    ),
)


def for_phase(phase: str) -> tuple[Technology, ...]:
    """Return technologies suitable for a given Dhamma phase."""
    return tuple(t for t in CATALOGUE if phase in t.phases)


def pick(phase: str, rng: random.Random) -> Technology:
    """Pick a technology for ``phase`` using ``rng`` (deterministic)."""
    candidates = for_phase(phase)
    if not candidates:
        return CATALOGUE[0]
    return rng.choice(candidates)


__all__ = ["CATALOGUE", "Technology", "for_phase", "pick"]
