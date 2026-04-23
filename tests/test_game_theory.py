"""Tests for game-theoretic utilities."""

from __future__ import annotations

import pytest

from lizard.game_theory import (
    best_response,
    cooperation_score,
    expected_payoff,
    nash_indicator,
)


def test_best_response_prisoners_dilemma() -> None:
    # Row payoff for prisoner's dilemma; actions: (0)=cooperate, (1)=defect
    payoff = [
        [3.0, 0.0],
        [5.0, 1.0],
    ]
    assert best_response(payoff, opponent_action=0) == 1
    assert best_response(payoff, opponent_action=1) == 1


def test_nash_indicator_prisoners_dilemma() -> None:
    row = [
        [3.0, 0.0],
        [5.0, 1.0],
    ]
    col = [
        [3.0, 5.0],
        [0.0, 1.0],
    ]
    assert nash_indicator(row, col) == [(1, 1)]


def test_nash_indicator_coordination_game() -> None:
    row = [
        [2.0, 0.0],
        [0.0, 1.0],
    ]
    col = [
        [2.0, 0.0],
        [0.0, 1.0],
    ]
    eqs = nash_indicator(row, col)
    assert (0, 0) in eqs
    assert (1, 1) in eqs


def test_expected_payoff_uniform_mix() -> None:
    payoff = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert expected_payoff(payoff, [0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_expected_payoff_validates_mixed_strategy() -> None:
    payoff = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
        expected_payoff(payoff, [0.6, 0.6], [0.5, 0.5])


def test_cooperation_score_bounds() -> None:
    assert cooperation_score([]) == 0.0
    assert cooperation_score([0.5, 0.5, 0.5]) == pytest.approx(1.0)
    polarised = cooperation_score([-1.0, 1.0, -1.0, 1.0])
    assert 0.0 <= polarised <= 1.0


def test_matrix_must_be_square() -> None:
    with pytest.raises(ValueError):
        best_response([[1.0, 0.0]], opponent_action=0)
