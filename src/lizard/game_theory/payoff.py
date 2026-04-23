"""Normal-form game utilities for two-player symmetric encounters.

Implementations follow standard game-theoretic definitions (von Neumann &
Morgenstern; Osborne, *A Course in Game Theory*). Payoff matrices are
``list[list[float]]`` of shape ``n_actions x n_actions`` indexed by
``(row_action, col_action)``.
"""

from __future__ import annotations

from collections.abc import Sequence

Matrix = Sequence[Sequence[float]]


def _validate_square(matrix: Matrix) -> int:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Payoff matrix must be a non-empty square matrix")
    return n


def best_response(payoff: Matrix, opponent_action: int) -> int:
    """Return the row action that maximises the row player's payoff."""
    n = _validate_square(payoff)
    if not 0 <= opponent_action < n:
        raise ValueError(f"opponent_action {opponent_action} out of range [0, {n})")
    best = 0
    best_value = payoff[0][opponent_action]
    for i in range(1, n):
        if payoff[i][opponent_action] > best_value:
            best_value = payoff[i][opponent_action]
            best = i
    return best


def nash_indicator(row_payoff: Matrix, col_payoff: Matrix) -> list[tuple[int, int]]:
    """Return every pure-strategy Nash equilibrium of the bimatrix game."""
    n = _validate_square(row_payoff)
    if len(col_payoff) != n or any(len(row) != n for row in col_payoff):
        raise ValueError("row and column payoff matrices must share shape")
    equilibria: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            row_ok = all(row_payoff[i][j] >= row_payoff[k][j] for k in range(n))
            col_ok = all(col_payoff[i][j] >= col_payoff[i][k] for k in range(n))
            if row_ok and col_ok:
                equilibria.append((i, j))
    return equilibria


def expected_payoff(
    payoff: Matrix,
    row_mixed: Sequence[float],
    col_mixed: Sequence[float],
) -> float:
    """Expected payoff for the row player under mixed strategies."""
    n = _validate_square(payoff)
    if len(row_mixed) != n or len(col_mixed) != n:
        raise ValueError("mixed strategy dimension must match payoff matrix")
    if abs(sum(row_mixed) - 1.0) > 1e-6 or abs(sum(col_mixed) - 1.0) > 1e-6:
        raise ValueError("mixed strategies must sum to 1")
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += row_mixed[i] * col_mixed[j] * payoff[i][j]
    return total


def cooperation_score(stances: Sequence[float]) -> float:
    """Return a value in ``[0, 1]`` measuring mutual alignment of stances.

    The score is ``1 - 2 * variance`` clamped to ``[0, 1]`` under the
    convention that stances live in ``[-1, 1]``. Perfectly aligned ensembles
    score 1; maximally polarised ensembles score 0.
    """
    if not stances:
        return 0.0
    mean = sum(stances) / len(stances)
    variance = sum((s - mean) ** 2 for s in stances) / len(stances)
    score = 1.0 - 2.0 * variance
    return max(0.0, min(1.0, score))


__all__ = [
    "Matrix",
    "best_response",
    "cooperation_score",
    "expected_payoff",
    "nash_indicator",
]
