"""
LLM client abstraction.

Provides a unified interface for interacting with Ollama locally.
All LLM interactions use the Ollama CLI with the gpt-oss:20b model.
"""

import asyncio
import ast
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from recruiting_agent_pollock.config import get_settings

logger = logging.getLogger(__name__)

# Default model for Ollama
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"


class Message(BaseModel):
    """A message in a conversation."""

    role: str = Field(..., description="Role of the speaker (system, user, assistant)")
    content: str = Field(..., description="Message content")


class LLMResponse(BaseModel):
    """Response from an LLM."""

    content: str = Field(..., description="Generated text content")
    finish_reason: str = Field(default="stop", description="Reason for completion")
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage information",
    )
    model: str = Field(default="", description="Model used for generation")
    raw_response: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw response from the API",
    )


class OllamaError(Exception):
    """Exception raised when Ollama CLI fails."""

    def __init__(self, message: str, return_code: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.return_code = return_code
        self.stderr = stderr


class LLMClientBase(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a chat completion.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional model-specific parameters.

        Returns:
            Generated response.
        """
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a text completion.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional model-specific parameters.

        Returns:
            Generated response.
        """
        ...


class LLMClient(LLMClientBase):
    """
    Ollama-based LLM client.

    Uses the Ollama CLI to run the gpt-oss:20b model locally.
    All generation happens through subprocess calls to `ollama run`.
    """

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 1,
        timeout: int = 120,
    ) -> None:
        """
        Initialize the Ollama LLM client.

        Args:
            model: Model name (defaults to gpt-oss:20b).
            max_retries: Number of retries on failure (default 1).
            timeout: Timeout in seconds for Ollama commands (default 120).
        """
        settings = get_settings()
        self._model = model or settings.llm_model_name or DEFAULT_OLLAMA_MODEL
        self._max_retries = max_retries
        self._timeout = timeout

        logger.info(f"Initialized Ollama LLM client with model: {self._model}")

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    def _build_prompt_from_messages(self, messages: list[Message]) -> str:
        """
        Build a single prompt string from a list of messages.

        Args:
            messages: List of conversation messages.

        Returns:
            Formatted prompt string.
        """
        prompt_parts: list[str] = []

        for msg in messages:
            role = msg.role.lower()
            content = msg.content.strip()

            if role == "system":
                prompt_parts.append(f"[SYSTEM]\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"[USER]\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"[ASSISTANT]\n{content}\n")
            else:
                prompt_parts.append(f"[{role.upper()}]\n{content}\n")

        # Add a final marker to indicate where the assistant should respond
        prompt_parts.append("[ASSISTANT]\n")

        return "\n".join(prompt_parts)

    def _run_ollama_sync(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """
        Run Ollama CLI synchronously with retry logic.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            top_p: Top-p sampling parameter.
            num_ctx: Context window size.

        Returns:
            The model's response text, stripped of whitespace.

        Raises:
            OllamaError: If Ollama fails after all retries.
        """
        # Build the command
        # Note: Ollama's `run` command accepts options via environment or modelfile
        # For runtime options, we use the API approach with `ollama run` and stdin
        cmd = ["ollama", "run", self._model]

        # Build options string for the prompt (Ollama doesn't support CLI flags directly)
        # We'll pass options through the prompt prefix for models that support it
        # or rely on modelfile defaults

        last_error: Exception | None = None
        attempts = 0

        while attempts <= self._max_retries:
            attempts += 1
            try:
                logger.debug(f"Running Ollama (attempt {attempts}): {' '.join(cmd)}")

                # Run ollama with the prompt via stdin
                process = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )

                if process.returncode != 0:
                    error_msg = process.stderr.strip() or f"Exit code: {process.returncode}"
                    logger.warning(f"Ollama failed (attempt {attempts}): {error_msg}")
                    last_error = OllamaError(
                        f"Ollama exited with code {process.returncode}",
                        return_code=process.returncode,
                        stderr=process.stderr,
                    )
                    continue

                # Success - return stripped output
                response = process.stdout.strip()
                logger.debug(f"Ollama response length: {len(response)} chars")
                return response

            except subprocess.TimeoutExpired as e:
                logger.warning(f"Ollama timed out after {self._timeout}s (attempt {attempts})")
                last_error = OllamaError(f"Ollama timed out after {self._timeout} seconds")

            except FileNotFoundError:
                error_msg = "Ollama CLI not found. Please install Ollama: https://ollama.ai"
                logger.error(error_msg)
                raise OllamaError(error_msg)

            except Exception as e:
                logger.warning(f"Ollama error (attempt {attempts}): {e}")
                last_error = OllamaError(str(e))

        # All retries exhausted
        raise last_error or OllamaError("Ollama failed after all retries")

    async def _run_ollama(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """
        Run Ollama CLI asynchronously.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            top_p: Top-p sampling parameter.
            num_ctx: Context window size.

        Returns:
            The model's response text, stripped of whitespace.
        """
        # Run the synchronous subprocess call in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._run_ollama_sync(prompt, temperature, max_tokens, top_p, num_ctx),
        )

    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a chat completion using Ollama.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters (top_p, num_ctx).

        Returns:
            Generated response.
        """
        # Build prompt from messages
        prompt = self._build_prompt_from_messages(messages)

        # Extract optional parameters
        top_p = kwargs.get("top_p")
        num_ctx = kwargs.get("num_ctx")

        try:
            response_text = await self._run_ollama(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                num_ctx=num_ctx,
            )

            return LLMResponse(
                content=response_text,
                finish_reason="stop",
                usage={},  # Ollama CLI doesn't provide token counts
                model=self._model,
                raw_response={"prompt": prompt, "response": response_text},
            )

        except OllamaError as e:
            logger.error(f"Ollama chat failed: {e}")
            return LLMResponse(
                content="",
                finish_reason="error",
                usage={},
                model=self._model,
                raw_response={"error": str(e)},
            )

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a text completion using Ollama.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters (top_p, num_ctx).

        Returns:
            Generated response.
        """
        # Extract optional parameters
        top_p = kwargs.get("top_p")
        num_ctx = kwargs.get("num_ctx")

        try:
            response_text = await self._run_ollama(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                num_ctx=num_ctx,
            )

            return LLMResponse(
                content=response_text,
                finish_reason="stop",
                usage={},
                model=self._model,
                raw_response={"prompt": prompt, "response": response_text},
            )

        except OllamaError as e:
            logger.error(f"Ollama completion failed: {e}")
            return LLMResponse(
                content="",
                finish_reason="error",
                usage={},
                model=self._model,
                raw_response={"error": str(e)},
            )

    async def chat_with_json(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a chat completion with JSON structured output.

        Args:
            messages: Conversation history.
            schema: Optional JSON schema for expected output.
            temperature: Sampling temperature (lower for more deterministic).
            **kwargs: Additional parameters.

        Returns:
            Parsed JSON response, or empty dict on error.
        """
        # Add schema instruction to messages if provided
        if schema:
            schema_instruction = Message(
                role="system",
                content=(
                    "You must respond with valid JSON only. No additional text or explanation. "
                    f"Your response must match this JSON schema: {json.dumps(schema)}"
                ),
            )
            augmented_messages = [schema_instruction] + messages
        else:
            # Just add a general JSON instruction
            json_instruction = Message(
                role="system",
                content="You must respond with valid JSON only. No additional text or explanation.",
            )
            augmented_messages = [json_instruction] + messages

        response = await self.chat(augmented_messages, temperature, **kwargs)

        if response.finish_reason == "error" or not response.content:
            logger.warning("JSON chat failed, returning empty dict")
            return {}

        # Try to parse JSON from response
        try:
            # Try to find JSON in the response (model might add extra text)
            content = response.content.strip()

            # Look for JSON object or array
            start_idx = content.find("{")
            if start_idx == -1:
                start_idx = content.find("[")

            if start_idx != -1:
                # Find matching closing bracket
                bracket_count = 0
                end_idx = start_idx
                open_bracket = content[start_idx]
                close_bracket = "}" if open_bracket == "{" else "]"

                for i, char in enumerate(content[start_idx:], start=start_idx):
                    if char == open_bracket:
                        bracket_count += 1
                    elif char == close_bracket:
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break

                json_str = content[start_idx:end_idx]

                parsed = self._parse_json_loose(json_str)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"items": parsed}

            # Fallback: try parsing the whole content
            parsed = self._parse_json_loose(content)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
            return {}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {response.content[:500]}")
            return {}

    def _fix_json_string(self, json_str: str) -> str:
        """
        Attempt to fix common JSON issues from LLM output.
        
        Args:
            json_str: Raw JSON string that may have issues.
            
        Returns:
            Cleaned JSON string.
        """
        import re

        if not json_str:
            return ""

        result = json_str.strip()

        # Strip common fenced blocks.
        result = re.sub(r"^```(?:json)?\s*", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\s*```$", "", result)

        # Normalize curly quotes.
        result = (
            result.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )

        # Remove trailing commas before closing braces/brackets.
        result = re.sub(r",(\s*[}\]])", r"\1", result)

        # Convert Python literals to JSON literals.
        result = re.sub(r"\bNone\b", "null", result)
        result = re.sub(r"\bTrue\b", "true", result)
        result = re.sub(r"\bFalse\b", "false", result)

        # Quote bare keys (very common: {foo: "bar"}).
        # Only targets object keys right after { or , to avoid mangling values.
        result = re.sub(
            r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)",
            r'\1"\2"\3',
            result,
        )

        # If it looks like a Python dict (single quotes used heavily), try a conservative swap.
        # We do this *after* key-quoting so we don't introduce bad escapes as often.
        if result.count("'") > 0 and result.count('"') == 0:
            result = result.replace("'", '"')

        return result

    def _coerce_to_json_types(self, obj: Any) -> Any:
        """Coerce a Python object to JSON-safe types.

        This is used after `ast.literal_eval` fallback to avoid leaking non-JSON
        Python sentinels (e.g., `Ellipsis`) into the rest of the system.
        """
        if obj is ...:
            return None
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._coerce_to_json_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._coerce_to_json_types(v) for v in obj]
        # Last resort: stringify unknown objects.
        return str(obj)

    def _parse_json_loose(self, raw: str) -> dict[str, Any] | list[Any] | None:
        """Parse JSON with best-effort repair.

        Returns a dict/list on success, else None.
        """
        if not raw:
            return None

        cleaned = self._fix_json_string(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: try parsing as a Python literal (handles single quotes, trailing commas).
        # Then round-trip through json for safety.
        try:
            obj = ast.literal_eval(raw.strip())
        except Exception:
            try:
                obj = ast.literal_eval(cleaned)
            except Exception:
                return None

        if not isinstance(obj, (dict, list, tuple, set)):
            return None

        coerced = self._coerce_to_json_types(obj)
        try:
            # Ensure it's JSON-serializable and normalize booleans/nulls.
            return json.loads(json.dumps(coerced))
        except Exception:
            return None

    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for text.

        Note: Ollama embeddings require a separate model (e.g., nomic-embed-text).
        This is a placeholder that returns an empty list.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (empty for now).
        """
        # Ollama embeddings would need: ollama run nomic-embed-text
        # For now, return empty - implement when embedding model is available
        logger.warning("Embeddings not implemented for Ollama CLI client")
        return []

    async def close(self) -> None:
        """Close the client and release resources."""
        # No persistent resources to close for subprocess-based client
        pass


# Convenience function for simple completions
async def generate(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """
    Simple function to generate text using Ollama.

    Args:
        prompt: The prompt to send to the model.
        model: Model name (defaults to gpt-oss:20b).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        Generated text, stripped of whitespace.
    """
    client = LLMClient(model=model)
    response = await client.complete(prompt, temperature, max_tokens)
    return response.content
