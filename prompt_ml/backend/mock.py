from __future__ import annotations

import json
from collections import deque
from typing import Any

from prompt_ml.backend.base import Message


class MockBackend:
    """
    Scriptable LLM backend for unit tests and local development.

    The backend works in two modes:

    **Scripted mode** — supply a list of responses at construction time.
    Each call to ``complete()`` pops the next response from the queue.
    When the queue is exhausted, ``complete()`` raises ``StopIteration``.

    **Echo mode** (default) — when no responses are queued, the backend
    returns the last user message verbatim.  Useful for smoke-testing
    pipeline wiring without worrying about realistic output.

    Usage::

        backend = MockBackend(responses=[
            '{"line1": "123 Main St", "city": "Springfield"}',
            '{"state": "IL", "zip_code": "62701"}',
        ])
        instruction = CollectAddress(backend=backend)
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._queue: deque[str] = deque(responses or [])
        self.call_log: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.call_log.append(messages)
        if self._queue:
            return self._queue.popleft()
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg["content"]
        return ""

    def enqueue(self, *responses: str) -> None:
        """Add more scripted responses at runtime."""
        self._queue.extend(responses)

    def enqueue_json(self, data: Any) -> None:
        """Convenience: serialise ``data`` and enqueue as the next response."""
        self._queue.append(json.dumps(data))

    @property
    def call_count(self) -> int:
        return len(self.call_log)

    def reset(self) -> None:
        self._queue.clear()
        self.call_log.clear()
