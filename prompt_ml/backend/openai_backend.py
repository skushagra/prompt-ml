from __future__ import annotations

from typing import Any

from prompt_ml.backend.base import Message


class OpenAIBackend:
    """
    LLM backend that calls the OpenAI Chat Completions API.

    Usage::

        from prompt_ml.backend.openai_backend import OpenAIBackend

        backend = OpenAIBackend(api_key="sk-...", model="gpt-4o-mini")
        CollectAddress.backend = backend
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAIBackend.\n"
                "Install it with:  pip install openai"
            ) from exc

        self.model = model
        self.temperature = temperature
        self._extra = kwargs
        self._client = OpenAI(api_key=api_key)

    def complete(self, messages: list[Message]) -> str:
        """Send ``messages`` to the model and return the reply text."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            **self._extra,
        )
        return response.choices[0].message.content or ""
