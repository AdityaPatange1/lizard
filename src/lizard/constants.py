"""Global constants shared across the SIMULA subsystem.

The constants here are deliberately terse and immutable: the rest of the
package reads from them so that behaviour is trivial to audit.
"""

from __future__ import annotations

from typing import Final

#: Inclusive lower bound on the number of agents a run may instantiate.
MIN_AGENTS: Final[int] = 5

#: Inclusive upper bound on the number of agents a run may instantiate.
MAX_AGENTS: Final[int] = 28

#: Default number of agents when the user does not override.
DEFAULT_AGENT_COUNT: Final[int] = 7

#: Default number of SIMULA steps per run.
DEFAULT_MAX_STEPS: Final[int] = 8

#: Default Boltzmann temperature (>0) used in thermodynamic action selection.
DEFAULT_TEMPERATURE: Final[float] = 1.0

#: Default local Ollama model id.
DEFAULT_OLLAMA_MODEL: Final[str] = "gemma3"

#: The four Dhamma phases (first half of the Noble Eightfold Path) executed
#: per step in the order given.
DHAMMA_PHASES: Final[tuple[str, ...]] = (
    "right_intention",
    "right_thought",
    "right_speech",
    "right_action",
)

#: Archetypal roles drawn from the "world technologies" catalogue; the engine
#: draws N roles from this pool (with replacement allowed) to seed an
#: N-agent ensemble.
AGENT_ROLES: Final[tuple[str, ...]] = (
    "analyst",
    "skeptic",
    "mediator",
    "archivist",
    "ethicist",
    "strategist",
    "empath",
    "contrarian",
    "realist",
    "planner",
    "historian",
    "engineer",
    "jurist",
    "investigator",
)
