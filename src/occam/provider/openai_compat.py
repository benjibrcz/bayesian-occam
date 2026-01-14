"""OpenAI-compatible API client for Hyperbolic and similar providers."""

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CompletionResult:
    """Result from a chat completion request."""

    text: str
    raw: dict[str, Any]
    usage: dict[str, int] | None = None


class OpenAICompatClient:
    """Minimal OpenAI-compatible chat completions client.

    Works with Hyperbolic and other OpenAI-compatible providers.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize the client.

        Args:
            base_url: Base URL for the API. Defaults to HYPERBOLIC_BASE_URL env var
                     or https://api.hyperbolic.xyz/v1.
            api_key: API key. Defaults to HYPERBOLIC_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        self.base_url = (
            base_url
            or os.environ.get("HYPERBOLIC_BASE_URL")
            or "https://api.hyperbolic.xyz/v1"
        )
        self.api_key = api_key or os.environ.get("HYPERBOLIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Set HYPERBOLIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> CompletionResult:
        """Send a chat completion request.

        Args:
            model: Model identifier to use.
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens: Maximum tokens in the response.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            CompletionResult with text, raw response, and usage info.

        Raises:
            httpx.HTTPError: If the request fails.
            ValueError: If the response format is unexpected.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()

        # Extract the assistant's message
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format: {data}") from e

        # Extract usage if present
        usage = data.get("usage")

        return CompletionResult(text=text, raw=data, usage=usage)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "OpenAICompatClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
