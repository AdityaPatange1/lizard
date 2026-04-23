"""Classical search techniques used to explore deliberation plans.

The module provides Breadth-First, Depth-First, Uniform-Cost, A\\*, and
Beam search over a generic state graph, following Russell & Norvig's
*Artificial Intelligence: A Modern Approach* (chapters 3-4).
"""

from lizard.search.heuristics import (
    SearchNode,
    SearchProblem,
    astar_search,
    beam_search,
    breadth_first_search,
    depth_first_search,
    uniform_cost_search,
)

__all__ = [
    "SearchNode",
    "SearchProblem",
    "astar_search",
    "beam_search",
    "breadth_first_search",
    "depth_first_search",
    "uniform_cost_search",
]
