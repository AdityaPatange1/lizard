"""Tests for the classical search algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lizard.search import (
    astar_search,
    beam_search,
    breadth_first_search,
    depth_first_search,
    uniform_cost_search,
)


@dataclass
class GridProblem:
    """Tiny 3x3 grid problem used in multiple tests."""

    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (2, 2)
    size: int = 3

    def initial_state(self) -> tuple[int, int]:
        return self.start

    def is_goal(self, state: tuple[int, int]) -> bool:
        return state == self.goal

    def actions(self, state: tuple[int, int]) -> list[str]:
        return ["up", "down", "left", "right"]

    def result(self, state: tuple[int, int], action: str) -> tuple[int, int]:
        x, y = state
        if action == "up":
            y = max(0, y - 1)
        elif action == "down":
            y = min(self.size - 1, y + 1)
        elif action == "left":
            x = max(0, x - 1)
        elif action == "right":
            x = min(self.size - 1, x + 1)
        return (x, y)

    def step_cost(
        self,
        state: tuple[int, int],
        action: str,
        next_state: tuple[int, int],
    ) -> float:
        return 1.0 if state != next_state else 0.0


def manhattan(state: tuple[int, int]) -> float:
    return abs(state[0] - 2) + abs(state[1] - 2)


def test_bfs_finds_goal() -> None:
    path = breadth_first_search(GridProblem())
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # 4 moves + start


def test_dfs_finds_goal_within_depth_limit() -> None:
    path = depth_first_search(GridProblem(), max_depth=16)
    assert path is not None
    assert path[-1] == (2, 2)


def test_dfs_respects_depth_limit() -> None:
    assert depth_first_search(GridProblem(), max_depth=1) is None


def test_ucs_matches_bfs_on_unit_costs() -> None:
    ucs = uniform_cost_search(GridProblem())
    assert ucs is not None
    assert ucs[-1] == (2, 2)
    assert len(ucs) == 5


def test_astar_with_manhattan_is_optimal() -> None:
    path = astar_search(GridProblem(), manhattan)
    assert path is not None
    assert len(path) == 5


def test_beam_search_finds_goal() -> None:
    path = beam_search(GridProblem(), manhattan, beam_width=2)
    assert path is not None
    assert path[-1] == (2, 2)


def test_beam_search_rejects_zero_width() -> None:
    with pytest.raises(ValueError):
        beam_search(GridProblem(), manhattan, beam_width=0)


def test_search_handles_start_is_goal() -> None:
    problem = GridProblem(start=(2, 2))
    assert breadth_first_search(problem) == [(2, 2)]
    assert depth_first_search(problem) == [(2, 2)]
    assert astar_search(problem, manhattan) == [(2, 2)]
