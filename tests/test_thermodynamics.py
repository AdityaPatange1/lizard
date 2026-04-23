"""Tests for the thermodynamics module."""

from __future__ import annotations

import math
import random

import pytest

from lizard.thermodynamics import (
    boltzmann_select,
    free_energy,
    shannon_entropy,
    softmax,
)


def test_softmax_sums_to_one() -> None:
    probs = softmax([1.0, 2.0, 3.0], temperature=1.0)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)


def test_softmax_higher_temperature_flattens() -> None:
    hot = softmax([1.0, 2.0, 3.0], temperature=10.0)
    cold = softmax([1.0, 2.0, 3.0], temperature=0.1)
    assert max(hot) < max(cold)


def test_softmax_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0.0)


def test_shannon_entropy_uniform() -> None:
    probs = [0.25, 0.25, 0.25, 0.25]
    assert shannon_entropy(probs) == pytest.approx(2.0)


def test_shannon_entropy_point_mass() -> None:
    assert shannon_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_shannon_entropy_rejects_bad_distribution() -> None:
    with pytest.raises(ValueError):
        shannon_entropy([0.5, 0.6])


def test_boltzmann_select_distribution() -> None:
    rng = random.Random(0)
    counts = [0, 0, 0]
    for _ in range(2000):
        idx = boltzmann_select([1.0, 5.0, 1.0], temperature=1.0, rng=rng)
        counts[idx] += 1
    # The middle action is by far the most likely.
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_free_energy_reduces_to_single_state() -> None:
    # With a single-state system, F = E.
    assert free_energy([3.0], temperature=1.0) == pytest.approx(3.0)


def test_free_energy_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        free_energy([1.0, 2.0], temperature=0.0)
