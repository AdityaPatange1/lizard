"""A SIMULA *scenario*: the life situation an ensemble deliberates upon."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """A prompt plus derived framing metadata.

    The ``focus`` line is what the LLM/fallback keys off; it is a short,
    single-sentence extraction from the user's prompt.
    """

    prompt: str
    focus: str

    @classmethod
    def from_prompt(cls, prompt: str) -> Scenario:
        cleaned = " ".join(prompt.strip().split())
        if not cleaned:
            raise ValueError("scenario prompt must not be empty")
        focus = cleaned if len(cleaned) <= 180 else cleaned[:177] + "..."
        return cls(prompt=cleaned, focus=focus)


__all__ = ["Scenario"]
