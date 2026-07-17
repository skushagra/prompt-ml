from __future__ import annotations

from typing import Protocol, runtime_checkable


Message = dict[str, str]


@runtime_checkable
class LLMBackend(Protocol):
    """
    Minimal protocol that every LLM backend adapter must satisfy.

    A backend is responsible for exactly one thing: given a list of chat
    messages, return the model's reply as a plain string.  All higher-level
    concerns (prompt construction, JSON parsing, retries) live above this
    layer in the instruction classes.

    To add a new backend (OpenAI, Anthropic, local Ollama, etc.) implement
    this protocol and pass an instance to the instruction at construction time
    or set it on the class::

        DataCollectionInstruction.backend = OpenAIBackend(model="gpt-4o")
    """

    def complete(self, messages: list[Message]) -> str:
        """
        Send ``messages`` to the model and return its text response.

        Parameters
        ----------
        messages:
            Ordered list of chat messages, each a dict with ``"role"`` and
            ``"content"`` keys.  Follows the OpenAI chat format so adapters
            for other providers simply translate.

        Returns
        -------
        str
            The raw text of the model's reply.  The caller is responsible
            for any JSON parsing or post-processing.
        """
        ...
