"""Relearning routine backing ``lizard --load-system``.

The routine scans every historical run and report, aggregates empirical
priors (phase-conditioned payoff/entropy statistics, cooperation, the most
productive agent roles), and writes an updated SIMULA config snapshot to
both the filesystem and the SQLite ``configs`` table.

Philosophically: SIMULA crunches its outputs on every boot/load so it
"learns and relearns" from its own deliberations rather than relying on a
static prior.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lizard.config import LizardConfig
from lizard.constants import DHAMMA_PHASES
from lizard.storage.db import Database
from lizard.storage.files import ReportFiles

_LOG = logging.getLogger("lizard.learning")


@dataclass
class RelearnResult:
    """Structured summary of a relearning pass."""

    runs_seen: int
    reports_seen: int
    config_path: Path
    config_id: int
    payload: dict[str, Any] = field(default_factory=dict)


def relearn_system(
    *,
    config: LizardConfig,
    db: Database,
    files: ReportFiles,
) -> RelearnResult:
    """Examine all outputs, rebuild the SIMULA config, persist it."""
    runs = db.list_runs(limit=10_000)
    reports = db.list_reports(limit=10_000)

    phase_entropy: defaultdict[str, list[float]] = defaultdict(list)
    phase_payoff: defaultdict[str, list[float]] = defaultdict(list)
    role_counter: Counter[str] = Counter()
    stance_samples: list[float] = []

    for run in runs:
        steps = db.list_steps(run["id"])
        for step in steps:
            phase_entropy[step["phase"]].append(float(step["entropy"]))
            phase_payoff[step["phase"]].append(float(step["payoff"]))
        for agent in db.list_agents(run["id"]):
            role_counter[str(agent["role"])] += 1
            stance_samples.append(float(agent["stance"]))

    per_phase: dict[str, dict[str, float]] = {}
    for phase in DHAMMA_PHASES:
        ents = phase_entropy.get(phase, [])
        pays = phase_payoff.get(phase, [])
        per_phase[phase] = {
            "mean_entropy": _mean(ents),
            "stdev_entropy": _stdev(ents),
            "mean_payoff": _mean(pays),
            "stdev_payoff": _stdev(pays),
            "samples": float(len(ents)),
        }

    payload: dict[str, Any] = {
        "version": 1,
        "runs_seen": len(runs),
        "reports_seen": len(reports),
        "phase_priors": per_phase,
        "top_roles": [role for role, _ in role_counter.most_common(8)],
        "stance_mean": _mean(stance_samples),
        "stance_stdev": _stdev(stance_samples),
        "temperature": config.temperature,
        "agent_count": config.agent_count,
        "max_steps": config.max_steps,
    }

    _LOG.info(
        "Relearned over %d runs and %d reports; persisting new config.",
        len(runs),
        len(reports),
    )
    config_path = files.write_config(configs_dir=config.configs_dir, payload=payload)
    config_id = db.save_config(payload)
    return RelearnResult(
        runs_seen=len(runs),
        reports_seen=len(reports),
        config_path=config_path,
        config_id=config_id,
        payload=payload,
    )


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


__all__ = ["RelearnResult", "relearn_system"]
