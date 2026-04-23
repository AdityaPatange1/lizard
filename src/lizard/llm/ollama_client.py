"""Ollama chat client with a deterministic offline fallback.

Lizard is built to be fully functional even when no Ollama daemon is
running. If the :pypi:`ollama` package is missing or the daemon errors, the
client degrades to a pure-Python heuristic that still produces phase-shaped
text. This keeps tests deterministic and the CLI usable on flights.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - import behaviour exercised indirectly
    import ollama as _ollama
except Exception:  # pragma: no cover - missing optional dep
    _ollama = None  # type: ignore[assignment]

_LOG = logging.getLogger("lizard.llm")


def _ollama_disabled() -> bool:
    """Return True when Ollama should be skipped entirely.

    Honoured values of ``LIZARD_DISABLE_OLLAMA``: ``1``, ``true``, ``yes``
    (case-insensitive). Used by the test suite and by users who want a
    fast, deterministic run without spinning up a daemon.
    """
    raw = os.getenv("LIZARD_DISABLE_OLLAMA", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMResponse:
    """Envelope returned from :meth:`LLMClient.chat`."""

    text: str
    source: str  # "ollama" or "fallback"
    model: str


_FALLBACK_TEMPLATES: dict[str, tuple[str, ...]] = {
    "right_intention": (
        "I intend to understand the situation without prejudice, focusing on {focus}.",
        "My intention is to honour the dignity of every party while attending to {focus}.",
        "I aim to act from compassion and clarity, not reactivity, regarding {focus}.",
    ),
    "right_thought": (
        "Reasoning through {focus}, the salient uncertainties are incentives, timing, and harm.",
        "Considering {focus}, I weigh the entropy of the available evidence against plausible motives.",
        "On {focus}: the most charitable interpretation is that action was taken under incomplete information.",
    ),
    "right_speech": (
        "I would speak plainly: '{focus}' deserves a measured, non-accusatory summary of facts.",
        "My statement on {focus}: describe what is known, acknowledge what is not, avoid blame.",
        "I will communicate about {focus} with care, honesty, and no embellishment.",
    ),
    "right_action": (
        "Proposed action on {focus}: escalate to an impartial review and preserve records.",
        "Act on {focus} by documenting findings, notifying stakeholders, and preserving evidence.",
        "For {focus}, the proportionate action is de-escalation, disclosure, and corrective process.",
    ),
}


class LLMClient:
    """Tiny façade over :pypi:`ollama.chat` with an offline fallback."""

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._rng = rng or random.Random()
        self._client: Any | None = None
        if _ollama is not None and not _ollama_disabled():
            try:
                self._client = _ollama.Client(host=host) if host else _ollama.Client()
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.debug("Failed to construct ollama client: %s", exc)
                self._client = None

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        """Return True when an Ollama daemon answers a short ping."""
        if self._client is None:
            return False
        try:  # pragma: no cover - network side effect
            self._client.list()
            return True
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.debug("Ollama probe failed: %s", exc)
            return False

    def chat(
        self,
        *,
        system: str,
        user: str,
        phase: str,
        focus: str,
    ) -> LLMResponse:
        """Run a single chat turn or fall back to the deterministic stub."""
        if self._client is not None:
            try:  # pragma: no cover - network side effect
                response = self._client.chat(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    options={"temperature": 0.6},
                )
                text = self._extract_text(response)
                if text:
                    return LLMResponse(text=text.strip(), source="ollama", model=self._model)
            except Exception as exc:
                _LOG.debug("Ollama chat failed (%s); using fallback.", exc)
        return LLMResponse(
            text=self._fallback_text(phase=phase, focus=focus),
            source="fallback",
            model=self._model,
        )

    def _fallback_text(self, *, phase: str, focus: str) -> str:
        templates = _FALLBACK_TEMPLATES.get(phase)
        if not templates:
            return f"[{phase}] {focus}"
        template = self._rng.choice(templates)
        snippet = focus.strip() or "the present situation"
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        return template.format(focus=snippet)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull text out of an ollama response, tolerating both shapes."""
        try:
            message = response["message"]
            content = message["content"] if isinstance(message, dict) else message.content
        except (KeyError, AttributeError, TypeError):
            return ""
        return str(content or "")


__all__ = ["LLMClient", "LLMResponse"]
