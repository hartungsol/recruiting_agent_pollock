"""
Models module for LLM client abstraction.

Provides a unified interface for interacting with Ollama locally.
"""

from recruiting_agent_pollock.models.llm_client import (
    DEFAULT_OLLAMA_MODEL,
    LLMClient,
    LLMClientBase,
    LLMResponse,
    Message,
    OllamaError,
    generate,
)

__all__ = [
    "LLMClient",
    "LLMClientBase",
    "LLMResponse",
    "Message",
    "OllamaError",
    "DEFAULT_OLLAMA_MODEL",
    "generate",
]
