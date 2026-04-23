"""Dhamma report construction.

A *Dhamma report* is the structured output of a SIMULA run. It contains
the scenario, the ensemble, every phase utterance keyed by agent, and
aggregate measures (entropy, cooperation, free energy). The report
shape is stable across JSON and Markdown outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from lizard.constants import DHAMMA_PHASES
from lizard.game_theory import cooperation_score
from lizard.simula.agent import Agent
from lizard.simula.scenario import Scenario
from lizard.storage.memory import StepRecord
from lizard.thermodynamics import free_energy, shannon_entropy, softmax


@dataclass
class DhammaReport:
    """Serializable payload for a single SIMULA run."""

    created_at: str
    scenario: dict[str, str]
    agents: list[dict[str, Any]]
    steps: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Dhamma Report — {self.scenario['focus']}")
        lines.append("")
        lines.append(f"_Generated: {self.created_at}_")
        lines.append("")
        lines.append("## Scenario")
        lines.append("")
        lines.append(f"> {self.scenario['prompt']}")
        lines.append("")
        lines.append("## Ensemble")
        lines.append("")
        lines.append("| Name | Role | Stance | Energy |")
        lines.append("| --- | --- | ---: | ---: |")
        for agent in self.agents:
            lines.append(
                f"| {agent['name']} | {agent['role']} "
                f"| {agent['stance']:+.2f} | {agent['energy']:.2f} |"
            )
        lines.append("")
        lines.append("## Metrics")
        lines.append("")
        for key, value in sorted(self.metrics.items()):
            lines.append(f"- **{key}**: {value:.4f}")
        lines.append("")
        lines.append("## Deliberation")
        lines.append("")
        by_step: dict[int, list[dict[str, Any]]] = {}
        for step in self.steps:
            by_step.setdefault(int(step["step_index"]), []).append(step)
        for idx in sorted(by_step):
            lines.append(f"### Step {idx + 1}")
            lines.append("")
            for phase in DHAMMA_PHASES:
                rows = [s for s in by_step[idx] if s["phase"] == phase]
                if not rows:
                    continue
                lines.append(f"**{_phase_title(phase)}**")
                lines.append("")
                for row in rows:
                    lines.append(f"- _{row['agent_name']}_ ({row['agent_role']}): {row['content']}")
                lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(self.summary or "_No summary generated._")
        lines.append("")
        return "\n".join(lines)


def _phase_title(phase: str) -> str:
    return phase.replace("_", " ").title()


def build_report(
    *,
    scenario: Scenario,
    agents: list[Agent],
    records: list[StepRecord],
    temperature: float,
) -> DhammaReport:
    """Aggregate step records into a :class:`DhammaReport` with metrics."""
    entropies = [r.entropy for r in records]
    payoffs = [r.payoff for r in records]
    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    mean_payoff = sum(payoffs) / len(payoffs) if payoffs else 0.0
    cooperation = cooperation_score([a.stance for a in agents])

    if payoffs:
        probs = softmax(payoffs, temperature=max(temperature, 1e-3))
        payoff_entropy = shannon_entropy(probs)
        fe = free_energy(payoffs, temperature=max(temperature, 1e-3))
    else:
        payoff_entropy = 0.0
        fe = 0.0

    metrics: dict[str, float] = {
        "mean_step_entropy": mean_entropy,
        "mean_payoff": mean_payoff,
        "cooperation": cooperation,
        "payoff_entropy": payoff_entropy,
        "free_energy": fe,
    }
    summary = _compose_summary(metrics, scenario=scenario)
    return DhammaReport(
        created_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        scenario={"prompt": scenario.prompt, "focus": scenario.focus},
        agents=[
            {
                "name": a.name,
                "role": a.role,
                "stance": a.stance,
                "energy": a.energy,
            }
            for a in agents
        ],
        steps=[
            {
                "step_index": r.step_index,
                "phase": r.phase,
                "agent_name": r.agent_name,
                "agent_role": r.agent_role,
                "content": r.content,
                "entropy": r.entropy,
                "payoff": r.payoff,
            }
            for r in records
        ],
        metrics=metrics,
        summary=summary,
    )


def _compose_summary(metrics: dict[str, float], *, scenario: Scenario) -> str:
    coop = metrics["cooperation"]
    if coop > 0.75:
        tone = "The ensemble converged with high cooperation"
    elif coop > 0.45:
        tone = "The ensemble found partial agreement"
    else:
        tone = "The ensemble remained polarised"
    fe = metrics["free_energy"]
    return (
        f"{tone} on the situation '{scenario.focus}'. "
        f"Mean step entropy was {metrics['mean_step_entropy']:.3f}, "
        f"mean payoff {metrics['mean_payoff']:.3f}, "
        f"and Helmholtz free energy {fe:.3f}. "
        "Right Intention, Thought, Speech and Action have been recorded above."
    )


__all__ = ["DhammaReport", "build_report"]
