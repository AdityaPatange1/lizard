"""Entropy, softmax, free energy, and Boltzmann action selection.

References
----------
* Shannon, C.E., *A Mathematical Theory of Communication* (1948).
* Sutton, R. and Barto, A., *Reinforcement Learning: An Introduction*,
  2nd ed., §2.3 on Boltzmann exploration.
* Friston, K., *The free-energy principle: a unified brain theory?* (2010).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def shannon_entropy(probabilities: Sequence[float], *, base: float = 2.0) -> float:
    """Return the Shannon entropy of ``probabilities`` (must sum to 1).

    Zero-probability outcomes contribute zero under the usual convention
    ``0 * log 0 = 0``.
    """
    if not probabilities:
        return 0.0
    total = sum(probabilities)
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must be a positive distribution summing to 1, got {total}")
    if base <= 1.0:
        raise ValueError("logarithm base must be > 1")
    log_base = math.log(base)
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p) / log_base
    return entropy


def softmax(logits: Sequence[float], *, temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax with positive ``temperature``."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not logits:
        return []
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def boltzmann_select(
    values: Sequence[float],
    *,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> int:
    """Sample an action index with probability proportional to ``exp(v/T)``."""
    probs = softmax(values, temperature=temperature)
    rng = rng or random.Random()
    return _weighted_choice(probs, rng)


def free_energy(energies: Sequence[float], *, temperature: float = 1.0) -> float:
    """Helmholtz free energy ``-T * log Z`` with numerical stability."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not energies:
        return 0.0
    scaled = [-e / temperature for e in energies]
    m = max(scaled)
    z = sum(math.exp(x - m) for x in scaled)
    return -temperature * (m + math.log(z))


def _weighted_choice(weights: Sequence[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    threshold = rng.random() * total
    cumulative = 0.0
    for idx, w in enumerate(weights):
        cumulative += w
        if cumulative >= threshold:
            return idx
    return len(weights) - 1


__all__ = ["boltzmann_select", "free_energy", "shannon_entropy", "softmax"]
