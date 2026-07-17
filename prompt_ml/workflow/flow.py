from __future__ import annotations

from typing import Any

from prompt_ml.backend.base import LLMBackend
from prompt_ml.context import Context


class Flow:
    """
    Base class for a complete, runnable prompt-ml workflow.

    A ``Flow`` owns a :class:`~prompt_ml.context.Context` and a root
    workflow node (typically a :class:`~prompt_ml.workflow.Sequence`) that
    wires together all instructions and sub-flows.  It exposes the same
    interface as any other node (``execute``, ``is_complete``, ``reset``),
    so flows can be nested inside other ``Sequence`` or ``Branch`` nodes.

    Defining a flow
    ---------------
    Subclass ``Flow`` and implement :meth:`build`.  The method receives the
    shared context and backend and should return the root node::

        class DealershipFlow(Flow):
            def build(self, context: Context, backend: LLMBackend | None):
                return Sequence([
                    Greeting(),
                    Branch(
                        classifier=ClassifyIntent(context=context, backend=backend),
                        routes={
                            "service": Sequence([
                                CollectAddress(context=context, backend=backend),
                                BookService(context=context),
                                ServiceConfirmation(context=context),
                            ]),
                            "other": GenericFarewell(),
                        },
                        default="other",
                    ),
                ])

    Usage
    -----
    ::

        flow = DealershipFlow(backend=OpenAIBackend(api_key="sk-..."))

        # Opening message (greeting and any other auto-advance steps)
        print(flow.start())

        while not flow.is_complete():
            user_says = input("Caller: ")
            print("Agent:", flow.execute(user_says))

    Properties
    ----------
    ``flow.context``  — the shared ``Context`` (read-only)
    ``flow.history``  — list of ``{"role", "reply"}`` dicts (every turn)
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        context: Context | None = None,
    ) -> None:
        self._context: Context = context if context is not None else Context()
        self._backend: LLMBackend | None = backend
        self._history: list[dict[str, str]] = []
        self._root: Any = self.build(self._context, self._backend)

    def build(self, context: Context, backend: LLMBackend | None) -> Any:
        """
        Construct and return the root workflow node.

        Override this in subclasses.  Every instruction that needs the
        shared context or a backend must be instantiated here with those
        arguments passed explicitly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement build(context, backend)"
        )

    def start(self) -> str:
        """
        Trigger the opening of the flow.

        Runs all leading auto-advance steps (greeting, setup actions, etc.)
        and returns their combined output.  Call this once before the first
        user message.
        """
        return self.execute("")

    def execute(self, user_input: str) -> str:
        """Process one user turn and return the agent's response."""
        reply = self._root.execute(user_input)
        self._history.append({"user": user_input, "agent": reply})
        return reply

    def is_complete(self) -> bool:
        """True when the entire workflow has finished."""
        return self._root.is_complete()

    def reset(self) -> None:
        """Reset the flow and all its nodes to their initial state."""
        self._history.clear()
        self._root.reset()

    @property
    def context(self) -> Context:
        """The shared data context for this flow."""
        return self._context

    @property
    def history(self) -> list[dict[str, str]]:
        """Ordered log of every turn: ``[{"user": ..., "agent": ...}, ...]``."""
        return list(self._history)

    def __repr__(self) -> str:
        done = "complete" if self.is_complete() else "running"
        return f"{type(self).__name__}({done}, turns={len(self._history)})"
