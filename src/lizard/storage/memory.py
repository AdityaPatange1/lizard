"""In-memory step buffer used during a live SIMULA run."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepRecord:
    """A single Dhamma-phase utterance from a single agent in a single step."""

    step_index: int
    phase: str
    agent_name: str
    agent_role: str
    content: str
    entropy: float
    payoff: float


@dataclass
class StepBuffer:
    """Bounded in-memory buffer of :class:`StepRecord` values.

    The buffer is a :class:`collections.deque` under the hood so that
    ``push`` is O(1) and old records are evicted once ``maxlen`` is
    exceeded (defaults to :data:`None`, i.e. unbounded).
    """

    maxlen: int | None = None
    _records: deque[StepRecord] = field(init=False)

    def __post_init__(self) -> None:
        self._records = deque(maxlen=self.maxlen)

    def push(self, record: StepRecord) -> None:
        self._records.append(record)

    def extend(self, records: Iterable[StepRecord]) -> None:
        for r in records:
            self.push(r)

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def for_step(self, step_index: int) -> list[StepRecord]:
        return [r for r in self._records if r.step_index == step_index]

    def by_phase(self, phase: str) -> list[StepRecord]:
        return [r for r in self._records if r.phase == phase]

    def tail(self, n: int) -> list[StepRecord]:
        if n <= 0:
            return []
        return list(self._records)[-n:]


__all__ = ["StepBuffer", "StepRecord"]
