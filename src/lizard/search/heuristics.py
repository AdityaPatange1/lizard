"""Generic search primitives used by the SIMULA engine.

The algorithms here are intentionally small, pure-Python, dependency-free
implementations of the canonical graph-search algorithms. They operate on a
:class:`SearchProblem` protocol so the engine can swap state representations
trivially.

References
----------
* Russell, S. and Norvig, P., *Artificial Intelligence: A Modern Approach*,
  4th ed., chapters 3-4.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action")


class SearchProblem(Protocol, Generic[State, Action]):
    """Minimal problem interface used by every algorithm here."""

    def initial_state(self) -> State: ...

    def is_goal(self, state: State) -> bool: ...

    def actions(self, state: State) -> Iterable[Action]: ...

    def result(self, state: State, action: Action) -> State: ...

    def step_cost(self, state: State, action: Action, next_state: State) -> float: ...


@dataclass(order=True)
class SearchNode(Generic[State, Action]):
    """Node wrapper ordered by priority (used in PQ algorithms)."""

    priority: float
    state: State = field(compare=False)
    parent: SearchNode[State, Action] | None = field(default=None, compare=False)
    action: Action | None = field(default=None, compare=False)
    path_cost: float = field(default=0.0, compare=False)

    def path(self) -> list[State]:
        node: SearchNode[State, Action] | None = self
        out: list[State] = []
        while node is not None:
            out.append(node.state)
            node = node.parent
        out.reverse()
        return out


def _reconstruct(node: SearchNode[State, Action] | None) -> list[State] | None:
    return node.path() if node is not None else None


def breadth_first_search(
    problem: SearchProblem[State, Action],
) -> list[State] | None:
    """Classical BFS; optimal when all step costs are equal."""
    start = problem.initial_state()
    if problem.is_goal(start):
        return [start]
    frontier: deque[SearchNode[State, Action]] = deque([SearchNode(0.0, start)])
    explored: set[State] = {start}
    while frontier:
        node = frontier.popleft()
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state in explored:
                continue
            child = SearchNode(
                priority=node.path_cost + 1,
                state=child_state,
                parent=node,
                action=action,
                path_cost=node.path_cost + 1,
            )
            if problem.is_goal(child_state):
                return _reconstruct(child)
            explored.add(child_state)
            frontier.append(child)
    return None


def depth_first_search(
    problem: SearchProblem[State, Action],
    *,
    max_depth: int = 64,
) -> list[State] | None:
    """Depth-limited DFS with cycle detection along the current path."""
    start = problem.initial_state()
    if problem.is_goal(start):
        return [start]

    def recurse(
        node: SearchNode[State, Action],
        depth: int,
        on_path: set[State],
    ) -> SearchNode[State, Action] | None:
        if depth >= max_depth:
            return None
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state in on_path:
                continue
            child = SearchNode(
                priority=node.path_cost + 1,
                state=child_state,
                parent=node,
                action=action,
                path_cost=node.path_cost + 1,
            )
            if problem.is_goal(child_state):
                return child
            on_path.add(child_state)
            found = recurse(child, depth + 1, on_path)
            on_path.discard(child_state)
            if found is not None:
                return found
        return None

    root = SearchNode[State, Action](priority=0.0, state=start)
    result = recurse(root, 0, {start})
    return _reconstruct(result)


def uniform_cost_search(
    problem: SearchProblem[State, Action],
) -> list[State] | None:
    """Dijkstra-equivalent UCS."""
    start = problem.initial_state()
    frontier: list[SearchNode[State, Action]] = []
    counter = 0
    heapq.heappush(frontier, SearchNode(0.0, start))
    best_cost: dict[State, float] = {start: 0.0}
    while frontier:
        node = heapq.heappop(frontier)
        if problem.is_goal(node.state):
            return _reconstruct(node)
        if node.path_cost > best_cost.get(node.state, float("inf")):
            continue
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            cost = node.path_cost + problem.step_cost(node.state, action, child_state)
            if cost < best_cost.get(child_state, float("inf")):
                best_cost[child_state] = cost
                counter += 1
                child = SearchNode(
                    priority=cost,
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=cost,
                )
                heapq.heappush(frontier, child)
    return None


def astar_search(
    problem: SearchProblem[State, Action],
    heuristic: Callable[[State], float],
) -> list[State] | None:
    """A\\* search with an admissible ``heuristic``."""
    start = problem.initial_state()
    frontier: list[SearchNode[State, Action]] = []
    heapq.heappush(frontier, SearchNode(heuristic(start), start))
    best_cost: dict[State, float] = {start: 0.0}
    while frontier:
        node = heapq.heappop(frontier)
        if problem.is_goal(node.state):
            return _reconstruct(node)
        if node.path_cost > best_cost.get(node.state, float("inf")):
            continue
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            g = node.path_cost + problem.step_cost(node.state, action, child_state)
            if g < best_cost.get(child_state, float("inf")):
                best_cost[child_state] = g
                f = g + heuristic(child_state)
                heapq.heappush(
                    frontier,
                    SearchNode(
                        priority=f,
                        state=child_state,
                        parent=node,
                        action=action,
                        path_cost=g,
                    ),
                )
    return None


def beam_search(
    problem: SearchProblem[State, Action],
    heuristic: Callable[[State], float],
    *,
    beam_width: int = 4,
    max_steps: int = 32,
) -> list[State] | None:
    """Beam search keeping the ``beam_width`` most promising frontier nodes."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    start = problem.initial_state()
    if problem.is_goal(start):
        return [start]
    beam: list[SearchNode[State, Action]] = [SearchNode(heuristic(start), start)]
    seen: set[State] = {start}
    for _ in range(max_steps):
        candidates: list[SearchNode[State, Action]] = []
        for node in beam:
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                if child_state in seen:
                    continue
                cost = node.path_cost + problem.step_cost(node.state, action, child_state)
                child = SearchNode(
                    priority=cost + heuristic(child_state),
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=cost,
                )
                if problem.is_goal(child_state):
                    return _reconstruct(child)
                candidates.append(child)
                seen.add(child_state)
        if not candidates:
            return None
        candidates.sort(key=lambda n: n.priority)
        beam = candidates[:beam_width]
    return None


__all__ = [
    "SearchNode",
    "SearchProblem",
    "astar_search",
    "beam_search",
    "breadth_first_search",
    "depth_first_search",
    "uniform_cost_search",
]
