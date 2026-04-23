"""Tests for the CLI orchestration."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from lizard.cli import main


def test_no_args_prints_help(lizard_home) -> None:  # type: ignore[no-untyped-def]
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main([])
    assert exit_code == 0
    assert "lizard" in buf.getvalue().lower()


def test_run_simula_generates_report(lizard_home) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "--run-simula",
            "--prompt",
            "Two people getting arrested for mismanagement of records",
        ]
    )
    assert exit_code == 0
    reports_dir = lizard_home / "reports"
    assert any(p.suffix == ".json" for p in reports_dir.iterdir())
    assert any(p.suffix == ".md" for p in reports_dir.iterdir())


def test_view_stats_after_run(lizard_home) -> None:  # type: ignore[no-untyped-def]
    main(["--run-simula", "--prompt", "A small disagreement in a meeting"])
    assert main(["--view-stats"]) == 0


def test_load_system_writes_config(lizard_home) -> None:  # type: ignore[no-untyped-def]
    main(["--run-simula", "--prompt", "Neighbours arguing about a fence"])
    assert main(["--load-system"]) == 0
    configs = list((lizard_home / "configs").iterdir())
    assert configs, "expected at least one learned config file"
