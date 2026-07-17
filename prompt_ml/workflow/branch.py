from __future__ import annotations

from typing import Any, Callable


class Branch:
    """
    Route the conversation to one of several sub-flows based on a
    classification of the user's input.

    The classifier is any object with the instruction interface
    (``execute``, ``is_complete``, ``reset``).  In practice this is always
    a :class:`~prompt_ml.instructions.classify.ClassifyInstruction` instance.
    After the classifier runs, its result is matched against ``routes`` and
    the selected sub-flow receives all subsequent user turns.

    Parameters
    ----------
    classifier:
        An instruction (or workflow node) that, when executed, classifies
        the user's message and writes the label to the shared context.
        The label returned by ``execute()`` is used to select a route.
    routes:
        Mapping of label → step/sub-flow.  The selected sub-flow runs
        until it is complete.
    default:
        Label to fall back to when the classifier returns a label not in
        ``routes``.  If ``None`` and no match is found, a ``RuntimeError``
        is raised.

    How input is distributed
    -------------------------
    The user's message that triggers classification is **also forwarded** to
    the selected route's first step.  This means a caller who says "I need
    an oil change at 742 Main St" will have the message classified as
    "service" *and* passed to ``CollectAddress``, which may extract a partial
    address from it immediately.

    Example::

        Branch(
            classifier=ClassifyIntent(context=ctx, backend=backend),
            routes={
                "service": Sequence([CollectAddress(), BookService(), Farewell()]),
                "sales":   SalesFarewell(),
                "other":   GenericFarewell(),
            },
            default="other",
        )
    """

    def __init__(
        self,
        classifier: Any,
        routes: dict[str, Any],
        default: str | None = None,
    ) -> None:
        self._classifier = classifier
        self._routes = routes
        self._default = default
        self._selected_label: str | None = None
        self._active: Any | None = None

    def execute(self, user_input: str) -> str:
        if self._active is None:
            return self._classify_and_enter(user_input)
        return self._active.execute(user_input)

    def is_complete(self) -> bool:
        return self._active is not None and self._active.is_complete()

    def reset(self) -> None:
        self._selected_label = None
        self._active = None
        self._classifier.reset()
        for route in self._routes.values():
            route.reset()

    @property
    def selected_label(self) -> str | None:
        """The label chosen by the classifier, or None if not yet run."""
        return self._selected_label

    def _classify_and_enter(self, user_input: str) -> str:
        label = self._classifier.execute(user_input)
        self._selected_label = label

        route = self._routes.get(label)
        if route is None and self._default:
            route = self._routes.get(self._default)
        if route is None:
            available = list(self._routes)
            raise RuntimeError(
                f"Branch: classifier returned label {label!r} which has no "
                f"matching route.  Available routes: {available}"
            )

        self._active = route

        reply = self._active.execute(user_input)
        return reply

    def __repr__(self) -> str:
        if self._selected_label:
            return f"Branch(selected={self._selected_label!r})"
        return "Branch(pending classification)"
