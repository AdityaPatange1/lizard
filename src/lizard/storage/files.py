"""Disk writers for JSON and Markdown report artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ISO_FMT = "%Y-%m-%dT%H-%M-%SZ"


def _timestamp_slug() -> str:
    return datetime.now(tz=timezone.utc).strftime(_ISO_FMT)


@dataclass(frozen=True)
class ReportFiles:
    """Writer for the paired JSON + Markdown Dhamma report artifacts.

    The class is intentionally tiny so the engine never touches ``open``
    calls directly.
    """

    reports_dir: Path

    def write(self, *, title: str, data: dict[str, Any], markdown: str) -> tuple[Path, Path]:
        """Write a JSON and a Markdown file that share the same timestamp."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        slug = _timestamp_slug()
        safe_title = _slugify(title)
        json_path = self.reports_dir / f"{slug}-{safe_title}.json"
        md_path = self.reports_dir / f"{slug}-{safe_title}.md"
        json_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        return json_path, md_path

    def write_config(self, *, configs_dir: Path, payload: dict[str, Any]) -> Path:
        configs_dir.mkdir(parents=True, exist_ok=True)
        slug = _timestamp_slug()
        path = configs_dir / f"simula-{slug}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def list_reports(self) -> list[Path]:
        if not self.reports_dir.exists():
            return []
        return sorted(self.reports_dir.glob("*.json"))

    def read(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Report at {path} is not a JSON object")
        return data


def _slugify(value: str, *, max_length: int = 48) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    collapsed = "-".join(filter(None, cleaned.split("-")))
    return (collapsed or "report")[:max_length]


__all__ = ["ReportFiles"]
