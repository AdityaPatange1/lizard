"""SQLite persistence for SIMULA runs, agents, steps, reports and configs.

The database is intentionally tiny and normalised. Every write goes through
the :class:`Database` class so callers never touch raw SQL.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    prompt        TEXT    NOT NULL,
    agent_count   INTEGER NOT NULL,
    step_count    INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'running',
    summary       TEXT
);

CREATE TABLE IF NOT EXISTS agents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name       TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    stance     REAL    NOT NULL,
    energy     REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    phase      TEXT    NOT NULL,
    agent_id   INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    content    TEXT    NOT NULL,
    entropy    REAL    NOT NULL,
    payoff     REAL    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    created_at TEXT    NOT NULL,
    title      TEXT    NOT NULL,
    json_path  TEXT    NOT NULL,
    md_path    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS configs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT    NOT NULL,
    payload    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id);
CREATE INDEX IF NOT EXISTS idx_agents_run ON agents(run_id);
CREATE INDEX IF NOT EXISTS idx_reports_run ON reports(run_id);
"""


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thin, typed wrapper around a SQLite database."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------------------------------------------------- runs

    def create_run(self, *, prompt: str, agent_count: int) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (created_at, prompt, agent_count) VALUES (?, ?, ?)",
                (_utc_now(), prompt, agent_count),
            )
            run_id = cur.lastrowid
        assert run_id is not None
        return int(run_id)

    def finish_run(self, run_id: int, *, step_count: int, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status=?, step_count=?, summary=? WHERE id=?",
                ("complete", step_count, summary, run_id),
            )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- agents

    def add_agents(self, run_id: int, agents: Iterable[tuple[str, str, float, float]]) -> list[int]:
        """Insert agents and return the assigned ids in order.

        Each tuple is ``(name, role, stance, energy)``.
        """
        ids: list[int] = []
        with self._connect() as conn:
            for name, role, stance, energy in agents:
                cur = conn.execute(
                    "INSERT INTO agents (run_id, name, role, stance, energy) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, name, role, stance, energy),
                )
                assert cur.lastrowid is not None
                ids.append(int(cur.lastrowid))
        return ids

    def list_agents(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agents WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- steps

    def add_step(
        self,
        *,
        run_id: int,
        step_index: int,
        phase: str,
        agent_id: int,
        content: str,
        entropy: float,
        payoff: float,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO steps "
                "(run_id, step_index, phase, agent_id, content, entropy, payoff, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    step_index,
                    phase,
                    agent_id,
                    content,
                    entropy,
                    payoff,
                    _utc_now(),
                ),
            )
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def list_steps(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- reports

    def add_report(
        self,
        *,
        run_id: int | None,
        title: str,
        json_path: Path,
        md_path: Path,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO reports (run_id, created_at, title, json_path, md_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, _utc_now(), title, str(json_path), str(md_path)),
            )
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def list_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- configs

    def save_config(self, payload: dict[str, Any]) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO configs (created_at, payload) VALUES (?, ?)",
                (_utc_now(), json.dumps(payload, sort_keys=True)),
            )
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def latest_config(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM configs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        if isinstance(payload, dict):
            return payload
        return None

    def list_configs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at FROM configs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- stats

    def aggregate_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            runs = conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(SUM(step_count), 0) AS steps, "
                "COALESCE(AVG(agent_count), 0) AS avg_agents "
                "FROM runs"
            ).fetchone()
            reports = conn.execute("SELECT COUNT(*) AS n FROM reports").fetchone()
            configs = conn.execute("SELECT COUNT(*) AS n FROM configs").fetchone()
            entropy = conn.execute(
                "SELECT COALESCE(AVG(entropy), 0) AS avg_entropy, "
                "COALESCE(AVG(payoff), 0) AS avg_payoff "
                "FROM steps"
            ).fetchone()
        return {
            "runs": int(runs["n"]),
            "total_steps": int(runs["steps"]),
            "avg_agents": float(runs["avg_agents"] or 0.0),
            "reports": int(reports["n"]),
            "configs": int(configs["n"]),
            "avg_entropy": float(entropy["avg_entropy"] or 0.0),
            "avg_payoff": float(entropy["avg_payoff"] or 0.0),
        }


__all__ = ["Database"]
