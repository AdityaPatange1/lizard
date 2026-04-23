"""Tests for :mod:`lizard.storage`."""

from __future__ import annotations

import json

from lizard.storage import Database, ReportFiles, StepBuffer
from lizard.storage.memory import StepRecord


def test_database_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    db = Database(tmp_path / "lizard.db")
    run_id = db.create_run(prompt="Hello", agent_count=5)
    agent_ids = db.add_agents(
        run_id,
        [
            ("A", "analyst", 0.1, 0.8),
            ("B", "mediator", -0.2, 0.7),
        ],
    )
    assert len(agent_ids) == 2
    db.add_step(
        run_id=run_id,
        step_index=0,
        phase="right_intention",
        agent_id=agent_ids[0],
        content="intention",
        entropy=0.5,
        payoff=0.9,
    )
    db.finish_run(run_id, step_count=1, summary="ok")

    runs = db.list_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "complete"
    assert runs[0]["step_count"] == 1

    steps = db.list_steps(run_id)
    assert len(steps) == 1
    assert steps[0]["phase"] == "right_intention"

    stats = db.aggregate_stats()
    assert stats["runs"] == 1
    assert stats["total_steps"] == 1


def test_database_config_persistence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    db = Database(tmp_path / "lizard.db")
    payload = {"version": 1, "phase_priors": {"right_intention": {"mean_payoff": 0.5}}}
    cfg_id = db.save_config(payload)
    assert cfg_id > 0
    assert db.latest_config() == payload


def test_step_buffer_filters() -> None:
    buffer = StepBuffer(maxlen=10)
    for i in range(3):
        buffer.push(
            StepRecord(
                step_index=i,
                phase="right_intention",
                agent_name="A",
                agent_role="analyst",
                content="x",
                entropy=0.0,
                payoff=float(i),
            )
        )
    assert len(buffer) == 3
    assert len(buffer.for_step(1)) == 1
    assert len(buffer.by_phase("right_intention")) == 3
    assert buffer.tail(2)[0].step_index == 1


def test_report_files_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    files = ReportFiles(reports_dir=tmp_path / "reports")
    data = {"scenario": {"prompt": "hi", "focus": "hi"}, "agents": [], "steps": []}
    json_path, md_path = files.write(title="hi", data=data, markdown="# hi")
    assert json_path.exists()
    assert md_path.exists()
    assert files.read(json_path) == data
    listed = files.list_reports()
    assert json_path in listed
    # config writer
    cfg_path = files.write_config(configs_dir=tmp_path / "cfg", payload={"a": 1})
    assert json.loads(cfg_path.read_text()) == {"a": 1}
