from __future__ import annotations

from typing import Any, ClassVar

from prompt_ml.context import Context
from prompt_ml.fields import FieldDefinition


class InstructionMeta(type):
    """
    Metaclass that scans each Instruction subclass body at *definition* time,
    lifts every FieldDefinition attribute into ``cls._fields``, and removes it
    from the class namespace so it doesn't shadow the instance attribute that
    will hold the collected value.

    Inheritance is handled correctly: fields defined on a parent class are
    merged with (and can be overridden by) fields defined on child classes.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> "InstructionMeta":
        inherited: dict[str, FieldDefinition] = {}
        for base in bases:
            if hasattr(base, "_fields"):
                inherited.update(base._fields)

        own: dict[str, FieldDefinition] = {}
        keys_to_remove: list[str] = []
        for key, value in namespace.items():
            if isinstance(value, FieldDefinition):
                own[key] = value
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del namespace[key]

        namespace["_fields"] = {**inherited, **own}
        return super().__new__(mcs, name, bases, namespace)


class Instruction(metaclass=InstructionMeta):
    """
    Base class for all prompt-ml instructions.

    Workflow flags
    --------------
    ``auto_advance`` — when ``True``, a :class:`~prompt_ml.workflow.Sequence`
    runs this step immediately without waiting for user input (e.g.
    ``OutputInstruction``, ``ActionInstruction``).

    ``emit_output`` — when ``False``, a ``Sequence`` discards the string
    returned by ``execute()`` rather than surfacing it to the caller.  Used
    for internal routing steps such as ``ClassifyInstruction`` whose return
    value (the label) is not meant to be spoken to the user.

    An Instruction is a single, self-contained unit of work in a prompt
    program.  Subclasses declare their state as ``Field(...)`` class
    attributes; the metaclass collects these into ``_fields`` and
    ``__init__`` initialises per-instance slots for each one.

    Every instruction carries a :class:`~prompt_ml.context.Context` — the
    shared data store that flows through an entire workflow.  Instructions
    read from and write to the context so that downstream instructions have
    access to everything collected or computed so far.

    A fresh ``Context`` is created automatically if none is supplied.  Pass
    an explicit context when wiring instructions together in a workflow::

        ctx = Context()
        collect = CollectAddress(context=ctx)
        classify = ClassifyIntent(context=ctx)   # sees what collect wrote

    Lifecycle
    ---------
    1. Instantiate the instruction (optionally with a shared context).
    2. Call ``execute(user_input)`` for each turn of the conversation.
    3. ``validate()`` (called internally) indicates which required fields are
       still missing.
    4. When ``is_complete()`` returns ``True`` the instruction is done.
    5. Call ``reset()`` to clear all state and reuse the instance.
    """

    auto_advance: ClassVar[bool] = False

    emit_output: ClassVar[bool] = True

    def __init__(self, context: Context | None = None) -> None:
        for name, field_def in self._fields.items():
            object.__setattr__(self, name, field_def.default)

        self._context: Context = context if context is not None else Context()

    @property
    def context(self) -> Context:
        """The shared data context for this instruction."""
        return self._context


    def execute(self, user_input: str = "") -> str:
        """
        Process one turn of input and return the instruction's response.

        Subclasses implement this method.  Non-interactive instructions
        (e.g. ``ActionInstruction``, ``OutputInstruction``) may ignore
        ``user_input`` and can be called with no arguments.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement execute()"
        )

    def validate(self) -> list[str]:
        """
        Return the names of all *required* fields that have not yet been
        populated.  An empty list means the instruction is complete.
        """
        missing: list[str] = []
        for name, field_def in self._fields.items():
            if field_def.required and getattr(self, name) is None:
                missing.append(name)
        return missing

    def is_complete(self) -> bool:
        """True when all required fields have been collected."""
        return not self.validate()

    def reset(self) -> None:
        """Clear all field values back to their defaults."""
        for name, field_def in self._fields.items():
            object.__setattr__(self, name, field_def.default)

    def collected(self) -> dict[str, Any]:
        """Return a snapshot of all field values (including optional ones)."""
        return {name: getattr(self, name) for name in self._fields}

    def __repr__(self) -> str:
        state = ", ".join(
            f"{k}={getattr(self, k)!r}" for k in self._fields
        )
        return f"{type(self).__name__}({state})"
