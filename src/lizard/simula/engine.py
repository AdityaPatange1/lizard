"""SIMULA engine.

The engine orchestrates a run:

1. Build an ensemble of 5-28 agents from the config.
2. For each step, iterate the four Dhamma phases.
3. In each phase, each agent produces an utterance via the LLM (or
   deterministic fallback), flavoured by a world technology.
4. Thermodynamic entropy and a game-theoretic payoff are computed per
   utterance and streamed both to the SQLite database and to an
   in-memory :class:`StepBuffer`.
5. The run produces a :class:`DhammaReport` which is serialised to
   JSON + Markdown.

The engine is callback-driven so both the interactive CLI (press Enter
to advance) and the headless run share the same code path.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

from lizard.config import LizardConfig
from lizard.constants import DHAMMA_PHASES
from lizard.llm import LLMClient
from lizard.simula.agent import Agent, AgentEnsemble
from lizard.simula.dhamma import DhammaReport, build_report
from lizard.simula.prompts import SYSTEM_PROMPT, user_prompt
from lizard.simula.scenario import Scenario
from lizard.storage.db import Database
from lizard.storage.memory import StepBuffer, StepRecord
from lizard.thermodynamics import shannon_entropy, softmax
from lizard.world import pick as pick_technology

_LOG = logging.getLogger("lizard.simula.engine")

StepCallback = Callable[["StepResult"], None]


@dataclass(frozen=True)
class StepResult:
    """One completed (step, phase) group of records."""

    step_index: int
    phase: str
    records: tuple[StepRecord, ...]


class Simula:
    """SIMULA orchestrator."""

    def __init__(
        self,
        *,
        config: LizardConfig,
        db: Database,
        llm: LLMClient,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._llm = llm
        self._rng = rng or (
            random.Random(config.seed) if config.seed is not None else random.Random()
        )

    @property
    def rng(self) -> random.Random:
        return self._rng

    def spawn_ensemble(self, count: int | None = None) -> AgentEnsemble:
        """Create an ensemble of ``count`` (defaults to ``config.agent_count``)."""
        return AgentEnsemble.spawn(count or self._config.agent_count, rng=self._rng)

    def run(
        self,
        scenario: Scenario,
        *,
        ensemble: AgentEnsemble | None = None,
        step_callback: StepCallback | None = None,
    ) -> tuple[int, DhammaReport]:
        """Run a full simulation and return ``(run_id, report)``."""
        ensemble = ensemble or self.spawn_ensemble()
        run_id = self._db.create_run(prompt=scenario.prompt, agent_count=len(ensemble))
        agent_ids = self._db.add_agents(run_id, ensemble.as_rows())
        for agent, agent_id in zip(ensemble, agent_ids, strict=True):
            agent.id = agent_id

        buffer = StepBuffer()
        completed_steps = 0
        for step_index in range(self._config.max_steps):
            for phase in DHAMMA_PHASES:
                step_result = self._run_phase(
                    run_id=run_id,
                    scenario=scenario,
                    ensemble=ensemble,
                    step_index=step_index,
                    phase=phase,
                    buffer=buffer,
                )
                if step_callback is not None:
                    step_callback(step_result)
            completed_steps = step_index + 1

        report = build_report(
            scenario=scenario,
            agents=list(ensemble),
            records=list(buffer),
            temperature=self._config.temperature,
        )
        self._db.finish_run(run_id, step_count=completed_steps, summary=report.summary)
        return run_id, report

    def _run_phase(
        self,
        *,
        run_id: int,
        scenario: Scenario,
        ensemble: AgentEnsemble,
        step_index: int,
        phase: str,
        buffer: StepBuffer,
    ) -> StepResult:
        records: list[StepRecord] = []
        phase_payoffs: list[float] = []
        for agent in ensemble:
            tech = pick_technology(phase, self._rng)
            prompt = user_prompt(
                phase=phase,
                focus=scenario.focus,
                role=agent.role,
                technology=tech.name,
            )
            response = self._llm.chat(
                system=SYSTEM_PROMPT,
                user=prompt,
                phase=phase,
                focus=scenario.focus,
            )
            payoff = self._compute_payoff(agent=agent, phase=phase)
            phase_payoffs.append(payoff)
            entropy = self._compute_entropy(payoffs=phase_payoffs)

            assert agent.id is not None
            self._db.add_step(
                run_id=run_id,
                step_index=step_index,
                phase=phase,
                agent_id=agent.id,
                content=response.text,
                entropy=entropy,
                payoff=payoff,
            )
            record = StepRecord(
                step_index=step_index,
                phase=phase,
                agent_name=agent.name,
                agent_role=agent.role,
                content=response.text,
                entropy=entropy,
                payoff=payoff,
            )
            buffer.push(record)
            records.append(record)
        _LOG.debug("step=%s phase=%s produced %d utterances", step_index, phase, len(records))
        return StepResult(step_index=step_index, phase=phase, records=tuple(records))

    def _compute_payoff(self, *, agent: Agent, phase: str) -> float:
        """Combine the agent's energy, stance and a phase bias into a payoff.

        The payoff is bounded in roughly ``[-1, 2]``; higher is better.
        """
        phase_bias = {
            "right_intention": 0.2,
            "right_thought": 0.3,
            "right_speech": 0.25,
            "right_action": 0.35,
        }[phase]
        raw = 0.6 * agent.energy + 0.3 * (1.0 - abs(agent.stance)) + phase_bias
        jitter = self._rng.uniform(-0.1, 0.1)
        return round(raw + jitter, 4)

    @staticmethod
    def _compute_entropy(*, payoffs: list[float]) -> float:
        if len(payoffs) < 2:
            return 0.0
        probs = softmax(payoffs, temperature=1.0)
        return round(shannon_entropy(probs), 4)


__all__ = ["Simula", "StepResult"]
