from __future__ import annotations

from typing import ClassVar

from prompt_ml.backend.base import LLMBackend, Message
from prompt_ml.context import Context
from prompt_ml.instruction import Instruction


class OutputInstruction(Instruction):
    """
    A non-interactive instruction that produces a text output (a message
    to send to the user) without waiting for a response.

    Three modes — the developer picks exactly one per subclass:

    **Static mode** — a hardcoded string, zero LLM cost::

        class Greeting(OutputInstruction):
            text = "Thank you for calling Acme Auto. How can I help you today?"

    **Template mode** — an f-string interpolated against the shared Context.
    Use dotted keys from the context as placeholders::

        class AppointmentConfirmed(OutputInstruction):
            template = (
                "Great! Your {CollectVehicle.make} {CollectVehicle.model} "
                "is booked for service on {CollectAppointment.date} "
                "at {CollectAppointment.time}."
            )

    **LLM mode** — the model generates the output given a system prompt and
    (optionally) the full context as additional information.  Use when you
    want natural, personalised phrasing rather than a fixed script::

        class PersonalisedGreeting(OutputInstruction):
            system_prompt = (
                "Generate a warm, brief greeting for a caller at Acme Auto. "
                "Mention the caller's name if it is available in the context."
            )
            backend = OpenAIBackend(api_key="sk-...")

    Priority
    --------
    If more than one mode is configured, priority is:
    ``text`` > ``template`` > LLM (``system_prompt`` + ``backend``).

    The LLM is never called unless neither ``text`` nor ``template`` is set.
    """

    auto_advance: ClassVar[bool] = True
    emit_output:  ClassVar[bool] = True

    text: ClassVar[str] = ""

    template: ClassVar[str] = ""

    system_prompt: ClassVar[str] = ""
    backend: ClassVar[LLMBackend | None] = None

    def __init__(
        self,
        backend: LLMBackend | None = None,
        context: Context | None = None,
    ) -> None:
        super().__init__(context=context)
        self._backend: LLMBackend | None = backend

    def execute(self, user_input: str = "") -> str:
        """
        Produce and return the output text.

        ``user_input`` is accepted but ignored — ``OutputInstruction`` is
        non-interactive and does not depend on what the user said.
        """
        if self.text:
            return self.text

        if self.template:
            return self._render_template()

        return self._llm_generate()

    def _render_template(self) -> str:
        """
        Interpolate ``template`` against the shared Context snapshot.

        All context keys are available as ``{key}`` placeholders.  Dotted
        keys like ``CollectAddress.city`` must be written as
        ``{CollectAddress.city}`` — Python's str.format_map handles dots
        in keys via a helper mapping.
        """
        snapshot = self._context.snapshot()
        try:
            return self.template.format_map(_DotKeyMap(snapshot))
        except KeyError as exc:
            raise KeyError(
                f"{type(self).__name__}: template references missing context key {exc}.\n"
                f"Available keys: {sorted(snapshot)}"
            ) from exc

    def _llm_generate(self) -> str:
        """Generate output via LLM using system_prompt + context snapshot."""
        backend = self._backend or self.__class__.backend
        if backend is None:
            raise RuntimeError(
                f"{type(self).__name__}: LLM mode requires a backend.\n"
                "Set one at the class level or pass it to the constructor."
            )
        if not self.system_prompt:
            raise RuntimeError(
                f"{type(self).__name__}: LLM mode requires a system_prompt."
            )

        context_summary = "\n".join(
            f"  {k}: {v}" for k, v in sorted(self._context.snapshot().items())
        ) or "  (empty)"

        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    f"{self.system_prompt}\n\n"
                    f"Current context:\n{context_summary}"
                ),
            },
            {"role": "user", "content": "Generate the output now."},
        ]
        return backend.complete(messages)


class _DotKeyMap:
    """
    A mapping wrapper that allows ``str.format_map`` to resolve dotted keys
    like ``{CollectAddress.city}`` against a flat dict with dotted string
    keys like ``{"CollectAddress.city": "Springfield"}``.

    Python's format_map resolves ``{A.B}`` by calling ``mapping["A"]`` and
    then accessing ``.B`` on the result.  We return a ``_Proxy`` for unknown
    top-level keys so that attribute access chains work transparently.
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        if key in self._data:
            return str(self._data[key])
        return _DotProxy(key, self._data)


class _DotProxy:
    """
    Proxy returned by ``_DotKeyMap`` for a partial key.

    When ``str.format_map`` resolves ``{CollectAddress.city}`` it:
    1. Calls ``mapping["CollectAddress"]``  → returns ``_DotProxy("CollectAddress", data)``
    2. Accesses ``.city`` on that proxy    → ``__getattr__("city")``
    3. Calls ``__format__`` on the result  → returns the final string value

    Nested dotting (``{A.B.C}``) is handled by returning another proxy.
    """

    def __init__(self, prefix: str, data: dict) -> None:
        self._prefix = prefix
        self._data = data

    def __getattr__(self, name: str) -> object:
        full_key = f"{self._prefix}.{name}"
        if full_key in self._data:
            return str(self._data[full_key])
        return _DotProxy(full_key, self._data)

    def __format__(self, spec: str) -> str:
        if self._prefix not in self._data:
            raise KeyError(self._prefix)
        return format(str(self._data[self._prefix]), spec)

    def __str__(self) -> str:
        if self._prefix not in self._data:
            raise KeyError(self._prefix)
        return str(self._data[self._prefix])
