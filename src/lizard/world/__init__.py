"""Catalogue of 'world technologies' — tools and lenses SIMULA agents use.

A :class:`Technology` is a small affordance that biases an agent's
reasoning: e.g. *jurisprudence* favours legal framings, *forensics*
favours evidence-first framings, and so on. The catalogue is intentionally
small and curated; callers pick an affordance per phase to flavour prompts.
"""

from lizard.world.technology import CATALOGUE, Technology, for_phase, pick

__all__ = ["CATALOGUE", "Technology", "for_phase", "pick"]
