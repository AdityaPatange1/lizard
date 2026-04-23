"""End-to-end tests for the SIMULA engine."""

from __future__ import annotations

from lizard.constants import DHAMMA_PHASES
from lizard.simula import Scenario, Simula
from lizard.simula.dhamma import build_report
from lizard.storage import Database
from lizard.storage.memory import StepBuffer


def test_engine_runs_and_persists(simula: Simula, database: Database) -> None:
    scenario = Scenario.from_prompt("Two people getting arrested for mismanagement of records")
    run_id, report = simula.run(scenario)
    assert run_id > 0
    assert report.agents
    assert len(report.steps) == len(report.agents) * len(DHAMMA_PHASES) * 2
    assert "mean_step_entropy" in report.metrics
    assert database.list_steps(run_id)
    assert database.list_agents(run_id)


def test_report_markdown_contains_phases(simula: Simula) -> None:
    scenario = Scenario.from_prompt("A civic dispute about a public park")
    _, report = simula.run(scenario)
    md = report.to_markdown()
    for phase in DHAMMA_PHASES:
        assert phase.replace("_", " ").title() in md
    assert "Summary" in md


def test_step_callback_fires_for_each_phase(simula: Simula) -> None:
    phases: list[str] = []

    def cb(result) -> None:  # type: ignore[no-untyped-def]
        phases.append(result.phase)

    scenario = Scenario.from_prompt("Neighbours arguing about a shared fence")
    simula.run(scenario, step_callback=cb)
    assert len(phases) == 2 * len(DHAMMA_PHASES)


def test_build_report_on_empty_records_is_safe() -> None:
    scenario = Scenario.from_prompt("x")
    report = build_report(scenario=scenario, agents=[], records=list(StepBuffer()), temperature=1.0)
    assert report.scenario["prompt"] == "x"
    assert report.metrics["mean_step_entropy"] == 0.0


def test_scenario_rejects_empty_prompt() -> None:
    import pytest

    with pytest.raises(ValueError):
        Scenario.from_prompt("   ")
