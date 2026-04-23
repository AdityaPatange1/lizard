"""Command-line interface for Lizard SIMULA.

The CLI is deliberately argparse-driven (no click, no typer) so there is
zero surprising dependency, and every subcommand falls through to a single
orchestration function.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from lizard import __version__
from lizard.config import LizardConfig
from lizard.constants import DHAMMA_PHASES
from lizard.learning import relearn_system
from lizard.llm import LLMClient
from lizard.logging_utils import build_console, configure_logging
from lizard.reports import render_report, render_stats
from lizard.simula import Scenario, Simula, StepResult
from lizard.simula.dhamma import DhammaReport
from lizard.storage import Database, ReportFiles

_LOG = logging.getLogger("lizard.cli")

_PHASE_STYLE = {
    "right_intention": "dhamma.intention",
    "right_thought": "dhamma.thought",
    "right_speech": "dhamma.speech",
    "right_action": "dhamma.action",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lizard",
        description=(
            "SIMULA Agent — a terminal-first, multi-agent life-situation simulator "
            "following the first four steps of the Noble Eightfold Path."
        ),
    )
    parser.add_argument("--version", action="version", version=f"lizard {__version__}")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Boot interactively; ask for a prompt, then advance one step per Enter.",
    )
    group.add_argument(
        "--run-simula",
        action="store_true",
        help="Run a full simulation headlessly (requires --prompt).",
    )
    group.add_argument(
        "--view-stats",
        action="store_true",
        help="Print aggregate run statistics from the SIMULA database.",
    )
    group.add_argument(
        "--reports",
        action="store_true",
        help="List past reports and pretty-print a selected one.",
    )
    group.add_argument(
        "--generate-report",
        action="store_true",
        help="Prompt for a situation, run SIMULA, and dump JSON + Markdown.",
    )
    group.add_argument(
        "--load-system",
        action="store_true",
        help="Re-ingest all outputs and relearn the SIMULA config.",
    )

    parser.add_argument("--prompt", type=str, help="Situation prompt for --run-simula.")
    parser.add_argument(
        "--agents",
        type=int,
        help="Override the number of agents (must be in [5, 28]).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Override the number of SIMULA steps per run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = LizardConfig.from_env()
    if args.agents is not None:
        config = config.with_agent_count(args.agents)
    if args.steps is not None:
        config = config.with_max_steps(int(args.steps))

    config.ensure_dirs()
    console = build_console()
    configure_logging(level=config.log_level, console=console)
    db = Database(config.db_path)
    files = ReportFiles(reports_dir=config.reports_dir)
    llm = LLMClient(model=config.ollama_model, host=config.ollama_host)

    no_command = not any(
        [
            args.interactive,
            args.run_simula,
            args.view_stats,
            args.reports,
            args.generate_report,
            args.load_system,
        ]
    )
    if no_command:
        _print_banner(console, config, llm)
        parser.print_help()
        return 0

    if args.interactive:
        return _cmd_interactive(console, config=config, db=db, files=files, llm=llm)
    if args.run_simula:
        if not args.prompt:
            parser.error("--run-simula requires --prompt")
        return _cmd_run(
            console,
            config=config,
            db=db,
            files=files,
            llm=llm,
            prompt=args.prompt,
            stepwise=False,
        )
    if args.view_stats:
        return _cmd_stats(console, db=db)
    if args.reports:
        return _cmd_reports(console, db=db, files=files)
    if args.generate_report:
        return _cmd_generate_report(console, config=config, db=db, files=files, llm=llm)
    if args.load_system:
        return _cmd_load_system(console, config=config, db=db, files=files)

    return 0


# ------------------------------------------------------------------ commands


def _cmd_interactive(
    console: Console,
    *,
    config: LizardConfig,
    db: Database,
    files: ReportFiles,
    llm: LLMClient,
) -> int:
    _print_banner(console, config, llm)
    prompt = Prompt.ask("[bold cyan]Describe the life situation[/bold cyan]")
    if not prompt.strip():
        console.print("[red]Empty prompt; aborting.[/red]")
        return 2
    return _cmd_run(
        console,
        config=config,
        db=db,
        files=files,
        llm=llm,
        prompt=prompt,
        stepwise=True,
    )


def _cmd_run(
    console: Console,
    *,
    config: LizardConfig,
    db: Database,
    files: ReportFiles,
    llm: LLMClient,
    prompt: str,
    stepwise: bool,
) -> int:
    scenario = Scenario.from_prompt(prompt)
    simula = Simula(config=config, db=db, llm=llm)
    ensemble = simula.spawn_ensemble()
    _render_ensemble(console, ensemble)

    last_printed_step: list[int | None] = [None]

    def on_step(result: StepResult) -> None:
        if last_printed_step[0] != result.step_index:
            console.rule(f"[bold]Step {result.step_index + 1}/{config.max_steps}")
            last_printed_step[0] = result.step_index
            if stepwise and result.phase == DHAMMA_PHASES[0] and result.step_index > 0:
                _wait_for_enter(console)
        _render_phase(console, result)

    if stepwise:
        console.print(
            "[dim]Press [bold]Enter[/bold] to advance between SIMULA steps. Ctrl-C to abort.[/dim]"
        )

    try:
        run_id, report = simula.run(scenario, ensemble=ensemble, step_callback=on_step)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user.[/yellow]")
        return 130

    _persist_and_show(
        console,
        db=db,
        files=files,
        run_id=run_id,
        report=report,
    )
    return 0


def _cmd_generate_report(
    console: Console,
    *,
    config: LizardConfig,
    db: Database,
    files: ReportFiles,
    llm: LLMClient,
) -> int:
    _print_banner(console, config, llm)
    prompt = Prompt.ask("[bold cyan]Type the situation[/bold cyan]")
    if not prompt.strip():
        console.print("[red]Empty prompt; aborting.[/red]")
        return 2
    scenario = Scenario.from_prompt(prompt)
    simula = Simula(config=config, db=db, llm=llm)
    _, report = simula.run(scenario)
    _persist_and_show(console, db=db, files=files, run_id=None, report=report)
    return 0


def _cmd_stats(console: Console, *, db: Database) -> int:
    stats = db.aggregate_stats()
    render_stats(console, stats)
    return 0


def _cmd_reports(console: Console, *, db: Database, files: ReportFiles) -> int:
    reports = db.list_reports()
    if not reports:
        console.print("[yellow]No reports found yet. Run a simulation first.[/yellow]")
        return 0
    table = Table(title="Reports", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Created")
    table.add_column("Title")
    table.add_column("Run")
    for idx, row in enumerate(reports, start=1):
        table.add_row(
            str(idx),
            str(row["created_at"]),
            str(row["title"]),
            str(row["run_id"]) if row["run_id"] is not None else "—",
        )
    console.print(table)
    if not sys.stdin.isatty():
        return 0
    try:
        choice = IntPrompt.ask(
            "Show report #",
            default=1,
            show_default=True,
            choices=[str(i) for i in range(1, len(reports) + 1)],
        )
    except (EOFError, KeyboardInterrupt):
        return 0
    record = reports[choice - 1]
    try:
        data = files.read(_path_from(record["json_path"]))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Could not read report: {exc}[/red]")
        return 1
    render_report(console, data)
    return 0


def _cmd_load_system(
    console: Console,
    *,
    config: LizardConfig,
    db: Database,
    files: ReportFiles,
) -> int:
    console.print("[cyan]Relearning from all prior outputs...[/cyan]")
    result = relearn_system(config=config, db=db, files=files)
    console.print(
        Panel.fit(
            f"Runs examined: [bold]{result.runs_seen}[/bold]\n"
            f"Reports examined: [bold]{result.reports_seen}[/bold]\n"
            f"New config written: [green]{result.config_path}[/green]\n"
            f"Config id in DB: [bold]{result.config_id}[/bold]",
            title="Relearn complete",
            border_style="green",
        )
    )
    console.print_json(json.dumps(result.payload, sort_keys=True))
    return 0


# ------------------------------------------------------------------ helpers


def _print_banner(console: Console, config: LizardConfig, llm: LLMClient) -> None:
    console.print(
        Panel.fit(
            f"[bold]Lizard[/bold] v{__version__} — SIMULA Agent\n"
            f"[dim]Home:[/dim] {config.home}\n"
            f"[dim]Model:[/dim] {llm.model}\n"
            f"[dim]Agents:[/dim] {config.agent_count}  "
            f"[dim]Steps:[/dim] {config.max_steps}  "
            f"[dim]Temp:[/dim] {config.temperature}",
            border_style="magenta",
        )
    )


def _render_ensemble(console: Console, ensemble: Any) -> None:
    table = Table(title=f"Ensemble ({len(ensemble)} agents)", header_style="bold")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Stance", justify="right")
    table.add_column("Energy", justify="right")
    for agent in ensemble:
        table.add_row(
            agent.name,
            agent.role,
            f"{agent.stance:+.2f}",
            f"{agent.energy:.2f}",
        )
    console.print(table)


def _render_phase(console: Console, result: StepResult) -> None:
    style = _PHASE_STYLE.get(result.phase, "white")
    console.print(f"[{style}]▸ {_phase_title(result.phase)}[/]")
    for record in result.records:
        console.print(f"  [dim]{record.agent_name} ({record.agent_role})[/dim] {record.content}")


def _wait_for_enter(console: Console) -> None:
    if not sys.stdin.isatty():
        return
    try:
        console.input("[dim](press Enter for the next step)[/dim] ")
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt from None


def _persist_and_show(
    console: Console,
    *,
    db: Database,
    files: ReportFiles,
    run_id: int | None,
    report: DhammaReport,
) -> None:
    title = report.scenario["focus"]
    json_path, md_path = files.write(
        title=title,
        data=report.to_dict(),
        markdown=report.to_markdown(),
    )
    db.add_report(run_id=run_id, title=title, json_path=json_path, md_path=md_path)
    console.print(
        Panel.fit(
            f"[green]JSON:[/green] {json_path}\n[green]Markdown:[/green] {md_path}",
            title="Dhamma report saved",
            border_style="green",
        )
    )
    render_report(console, report.to_dict())


def _phase_title(phase: str) -> str:
    return phase.replace("_", " ").title()


def _path_from(value: Any) -> Any:
    from pathlib import Path

    return Path(str(value))


__all__ = ["build_parser", "main"]
