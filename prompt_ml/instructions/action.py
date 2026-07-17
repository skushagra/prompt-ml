from __future__ import annotations

from typing import Any, ClassVar

from prompt_ml.context import Context
from prompt_ml.instruction import Instruction


class ActionResult:
    """
    Returned by :meth:`ActionInstruction.run` to signal the outcome of
    an action and carry its output data.

    Use the class methods to create results::

        return ActionResult.ok(data={"booking_id": "B-1234"})
        return ActionResult.fail(error=e, message="Could not reach the API.")
    """

    def __init__(
        self,
        success: bool,
        data: dict[str, Any] | None = None,
        message: str = "",
        error: Exception | None = None,
    ) -> None:
        self.success = success
        self.data = data or {}
        self.message = message
        self.error = error

    @classmethod
    def ok(cls, data: dict[str, Any] | None = None, message: str = "") -> "ActionResult":
        """Create a successful result, optionally carrying output data."""
        return cls(success=True, data=data, message=message)

    @classmethod
    def fail(cls, message: str = "", error: Exception | None = None) -> "ActionResult":
        """Create a failure result."""
        return cls(success=False, message=message, error=error)

    def __repr__(self) -> str:
        status = "ok" if self.success else "fail"
        return f"ActionResult({status}, data={self.data!r}, message={self.message!r})"


class ActionInstruction(Instruction):
    """
    A non-interactive instruction that executes a Python callable as a
    side effect — calling an API, writing to a database, sending a
    notification, etc.

    ``ActionInstruction`` does not interact with the user.  It reads from
    the shared ``Context`` to get its inputs and writes its results back to
    the same context so downstream instructions have access to them.

    Defining an action
    ------------------
    Override :meth:`run` to implement the action.  It receives the shared
    context and returns an :class:`ActionResult`::

        class BookAppointment(ActionInstruction):
            context_namespace = "booking"

            def run(self) -> ActionResult:
                date  = self.context.get("CollectAppointment.date")
                time_ = self.context.get("CollectAppointment.time")
                vid   = self.context.get("CollectVehicle.vehicle_id")
                try:
                    booking = api.book(date=date, time=time_, vehicle_id=vid)
                    return ActionResult.ok(
                        data={"booking_id": booking.id},
                        message=f"Booked! Confirmation number: {booking.id}",
                    )
                except ApiError as e:
                    return ActionResult.fail(message="Booking failed.", error=e)

    Context output
    --------------
    On success, all keys in ``ActionResult.data`` are written to the shared
    Context under ``<context_namespace>.<key>``.  Downstream instructions
    (e.g. an ``OutputInstruction`` with a template) can then read these
    values::

        context["booking.booking_id"]   # → "B-1234"

    Hooks
    -----
    Override :meth:`on_success` or :meth:`on_failure` to customise the
    string returned by ``execute()`` in each case.  The default
    ``on_success`` returns an empty string (silent action); the default
    ``on_failure`` re-raises the error.

    ``max_retries``
    ---------------
    Set ``max_retries`` (class variable, default 0) to automatically retry
    the action on failure before calling ``on_failure``::

        class BookAppointment(ActionInstruction):
            max_retries = 2
    """

    auto_advance: ClassVar[bool] = True
    emit_output:  ClassVar[bool] = True

    context_namespace: ClassVar[str] = ""

    max_retries: ClassVar[int] = 0

    def __init__(self, context: Context | None = None) -> None:
        super().__init__(context=context)
        self._result: ActionResult | None = None

    def execute(self, user_input: str = "") -> str:
        """
        Run the action (with retries if configured) and return a response
        string.  Writes result data to the shared Context on success.
        """
        attempts = 0
        last_error: Exception | None = None

        while attempts <= self.max_retries:
            try:
                result = self.run()
            except Exception as exc:
                last_error = exc
                result = ActionResult.fail(message=str(exc), error=exc)

            self._result = result

            if result.success:
                self._write_to_context(result)
                return self.on_success(result)

            attempts += 1

        return self.on_failure(
            self._result or ActionResult.fail(error=last_error)
        )

    def run(self) -> ActionResult:
        """
        Implement the action here.

        Read inputs from ``self.context``, perform the side effect, and
        return an :class:`ActionResult`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement run()"
        )

    @property
    def result(self) -> ActionResult | None:
        """The result of the last ``execute()`` call, or None."""
        return self._result

    def is_complete(self) -> bool:
        return self._result is not None and self._result.success

    def reset(self) -> None:
        super().reset()
        self._result = None

    def on_success(self, result: ActionResult) -> str:
        """
        Return the string to emit after a successful action.

        Override for a custom confirmation message.  Default is silent
        (returns ``""``), which is appropriate when the next instruction
        in the workflow handles user-facing output.
        """
        return result.message

    def on_failure(self, result: ActionResult) -> str:
        """
        Called when all retry attempts have failed.

        Override to return a user-friendly error message instead of raising.
        Default re-raises the underlying exception if one is available.
        """
        if result.error:
            raise result.error
        raise RuntimeError(
            f"{type(self).__name__} failed: {result.message}"
        )

    def _write_to_context(self, result: ActionResult) -> None:
        ns = self.context_namespace or type(self).__name__
        for key, value in result.data.items():
            self._context.set(f"{ns}.{key}", value)
