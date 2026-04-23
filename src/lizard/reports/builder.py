"""Rich renderers for reports and aggregate stats."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from lizard.constants import DHAMMA_PHASES

_PHASE_STYLE = {
    "right_intention": "dhamma.intention",
    "right_thought": "dhamma.thought",
    "right_speech": "dhamma.speech",
    "right_action": "dhamma.action",
}


def render_report(console: Console, data: dict[str, Any]) -> None:
    """Pretty-print a Dhamma report dict to ``console``."""
    scenario = data.get("scenario", {})
    console.print(
        Panel.fit(
            f"[bold]{scenario.get('focus', '—')}[/bold]\n[dim]{scenario.get('prompt', '')}[/dim]",
            title="Dhamma Report",
            border_style="cyan",
        )
    )

    agents = data.get("agents", [])
    if agents:
        table = Table(title="Ensemble", header_style="bold")
        table.add_column("Name")
        table.add_column("Role")
        table.add_column("Stance", justify="right")
        table.add_column("Energy", justify="right")
        for agent in agents:
            table.add_row(
                str(agent["name"]),
                str(agent["role"]),
                f"{float(agent['stance']):+.2f}",
                f"{float(agent['energy']):.2f}",
            )
        console.print(table)

    metrics = data.get("metrics", {})
    if metrics:
        table = Table(title="Metrics", header_style="bold")
        table.add_column("Key")
        table.add_column("Value", justify="right")
        for key, value in sorted(metrics.items()):
            table.add_row(key, f"{float(value):.4f}")
        console.print(table)

    steps = data.get("steps", [])
    if steps:
        by_step: dict[int, list[dict[str, Any]]] = {}
        for step in steps:
            by_step.setdefault(int(step["step_index"]), []).append(step)
        for idx in sorted(by_step):
            console.rule(f"[bold]Step {idx + 1}")
            for phase in DHAMMA_PHASES:
                rows = [r for r in by_step[idx] if r["phase"] == phase]
                if not rows:
                    continue
                style = _PHASE_STYLE.get(phase, "white")
                console.print(f"[{style}]{_phase_title(phase)}[/]")
                for row in rows:
                    console.print(
                        f"  [dim]{row['agent_name']} ({row['agent_role']})[/dim] {row['content']}"
                    )

    summary = data.get("summary")
    if summary:
        console.print(Panel.fit(Markdown(summary), title="Summary", border_style="green"))


def render_stats(console: Console, stats: dict[str, Any]) -> None:
    """Pretty-print aggregate statistics."""
    table = Table(title="Lizard SIMULA — Statistics", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    rows = [
        ("Total runs", f"{stats.get('runs', 0)}"),
        ("Total steps", f"{stats.get('total_steps', 0)}"),
        ("Average agents per run", f"{float(stats.get('avg_agents', 0.0)):.2f}"),
        ("Reports stored", f"{stats.get('reports', 0)}"),
        ("Learned configs", f"{stats.get('configs', 0)}"),
        ("Average step entropy", f"{float(stats.get('avg_entropy', 0.0)):.4f}"),
        ("Average step payoff", f"{float(stats.get('avg_payoff', 0.0)):.4f}"),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


def _phase_title(phase: str) -> str:
    return phase.replace("_", " ").title()


__all__ = ["render_report", "render_stats"]
