"""LLM integration (Ollama, local ``gemma3``) with a deterministic fallback."""

from lizard.llm.ollama_client import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse"]
