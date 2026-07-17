from __future__ import annotations

from typing import Any, Callable

from prompt_ml.context import Context


class Loop:
    """
    Repeat a step until a termination condition is met or a maximum number
    of iterations is reached.

    After each iteration the step is reset and run again from the start.
    The loop is complete when either:

    - ``until(context)`` returns ``True``, or
    - the iteration count reaches ``max_iterations``.

    Parameters
    ----------
    step:
        The step (instruction or sub-flow) to repeat.
    until:
        A callable ``(context: Context) -> bool``.  Called after each
        iteration completes.  When it returns ``True`` the loop stops.
        If ``None``, only ``max_iterations`` controls termination.
    max_iterations:
        Hard upper bound on the number of iterations (default: 10).
        Prevents infinite loops when ``until`` never fires.
    context:
        The shared :class:`~prompt_ml.context.Context`.  Used when
        evaluating the ``until`` condition.

    Example — retry data collection until a valid booking exists::

        Loop(
            step=CollectAppointment(context=ctx, backend=backend),
            until=lambda ctx: ctx.get("booking.confirmed") is True,
            max_iterations=3,
            context=ctx,
        )
    """

    def __init__(
        self,
        step: Any,
        until: Callable[[Context], bool] | None = None,
        max_iterations: int = 10,
        context: Context | None = None,
    ) -> None:
        self._step = step
        self._until = until
        self._max = max_iterations
        self._context = context or Context()
        self._iteration: int = 0
        self._done: bool = False

    def execute(self, user_input: str) -> str:
        if self._done:
            return ""

        reply = self._step.execute(user_input)

        if self._step.is_complete():
            self._iteration += 1

            condition_met = (
                self._until is not None and self._until(self._context)
            )
            max_reached = self._iteration >= self._max

            if condition_met or max_reached:
                self._done = True
            else:
                self._step.reset()

        return reply

    def is_complete(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._done = False
        self._iteration = 0
        self._step.reset()

    @property
    def iteration(self) -> int:
        """Number of completed iterations so far."""
        return self._iteration

    def __repr__(self) -> str:
        status = "done" if self._done else f"iter {self._iteration}/{self._max}"
        return f"Loop({status})"
