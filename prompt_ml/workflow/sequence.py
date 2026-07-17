from __future__ import annotations

from typing import Any


class Sequence:
    """
    Run a list of steps in order, advancing automatically through steps that
    do not need user input (``auto_advance = True``).

    A step is any object with the interface::

        step.execute(user_input: str) -> str
        step.is_complete() -> bool
        step.reset() -> None

    Both :class:`~prompt_ml.instruction.Instruction` subclasses and other
    workflow primitives (``Branch``, ``Loop``, nested ``Sequence``) satisfy
    this interface and can be freely mixed.

    Auto-advance behaviour
    ----------------------
    Steps with ``auto_advance = True`` (``OutputInstruction``,
    ``ActionInstruction``) are run by the sequence itself — they do not need
    the caller to provide user input.  The sequence drains them immediately
    before and after every interactive step, collecting all of their output
    into the same response string.

    Example::

        flow = Sequence([
            Greeting(),           # auto_advance → runs at start
            CollectAddress(),     # interactive → waits for caller
            BookService(),        # auto_advance → runs when address complete
            FinalConfirmation(),  # auto_advance → runs immediately after
        ])

        opening = flow.execute("")          # "Thank you for calling Acme Auto."
        reply   = flow.execute("123 Main…") # ... address collection turns ...
        # After CollectAddress completes, BookService + FinalConfirmation fire
        # automatically and their outputs are appended to the same reply.
    """

    def __init__(self, steps: list[Any]) -> None:
        self._steps = list(steps)
        self._idx: int = 0

    def execute(self, user_input: str) -> str:
        parts: list[str] = []

        self._drain(parts)

        if self.is_complete():
            return _join(parts)

        if not user_input or not user_input.strip():
            return _join(parts)

        step = self._steps[self._idx]
        reply = step.execute(user_input)
        if _emits(step) and reply:
            parts.append(reply)

        if step.is_complete():
            self._idx += 1
            self._drain(parts)

        return _join(parts)

    def is_complete(self) -> bool:
        return self._idx >= len(self._steps)

    def reset(self) -> None:
        self._idx = 0
        for step in self._steps:
            step.reset()

    def _drain(self, parts: list[str]) -> None:
        """
        Run all consecutive auto-advance steps starting from the current
        position.  Stops at the first interactive step or when the
        sequence is exhausted.
        """
        while self._idx < len(self._steps):
            step = self._steps[self._idx]
            if not getattr(step, "auto_advance", False):
                break
            reply = step.execute("")
            if _emits(step) and reply:
                parts.append(reply)
            if step.is_complete():
                self._idx += 1
            else:
                break

    def __repr__(self) -> str:
        done = self._idx
        total = len(self._steps)
        return f"Sequence(step {done}/{total})"


def _emits(step: Any) -> bool:
    return getattr(step, "emit_output", True)


def _join(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())
