from __future__ import annotations

import json
import re
from enum import Enum, auto
from typing import ClassVar

from prompt_ml.backend.base import LLMBackend, Message
from prompt_ml.context import Context
from prompt_ml.instruction import Instruction


class _Phase(Enum):
    COLLECTING = auto()
    CONFIRMING = auto()
    COMPLETE   = auto()


class DataCollectionInstruction(Instruction):
    """
    An Instruction that collects structured data from a user over a
    multi-turn conversation, then optionally confirms the collected data
    before marking itself complete.

    Execution loop (one call to ``execute()`` per user turn)
    --------------------------------------------------------
    **Collecting phase:**

    1. Append the user message to ``_history``.
    2. ``_extract(user_input)`` — LLM call: return a JSON dict of any field
       values found in the user's message.
    3. ``_apply(extracted)`` — write values onto ``self`` *and* into the
       shared ``Context`` (namespaced as ``<ClassName>.<field_name>``).
    4. ``validate()`` — check which required fields are still empty.
    5a. Fields still missing → ``_ask_for(field)`` — LLM call to generate a
        natural follow-up question.
    5b. All required fields filled → if ``auto_confirm`` is True, transition
        to the **confirming phase**; otherwise call ``on_complete()``.

    **Confirming phase:**

    1. Generate a natural summary of collected data and ask the user to
       confirm.
    2. ``_classify_confirmation(user_input)`` — LLM call that returns
       ``"confirmed"`` or ``"correction"``.
    3. If ``"confirmed"`` → call ``on_complete()`` and mark complete.
    4. If ``"correction"`` → hand the correction back to the extraction
       step so the user can supply amended values.

    Configuration
    -------------
    ``backend`` — set at class level or pass to ``__init__``::

        CollectAddress.backend = OpenAIBackend(api_key="sk-...")

    ``auto_confirm`` — set ``False`` to skip the confirmation phase::

        class QuickCollect(DataCollectionInstruction):
            auto_confirm = False

    ``system_context`` — inject a persona or domain description::

        class DealershipInstruction(DataCollectionInstruction):
            system_context = "You are a friendly Acme Auto phone assistant."

    ``context_namespace`` — the prefix used when writing to the shared
    Context.  Defaults to the class name.  Override to avoid collisions::

        class CollectAddress(DealershipInstruction):
            context_namespace = "address"
    """

    backend: ClassVar[LLMBackend | None] = None

    system_context: ClassVar[str] = (
        "You are a helpful assistant collecting information from a user."
    )

    auto_confirm: ClassVar[bool] = True

    context_namespace: ClassVar[str] = ""

    def __init__(
        self,
        backend: LLMBackend | None = None,
        context: Context | None = None,
    ) -> None:
        super().__init__(context=context)
        self._backend: LLMBackend | None = backend
        self._history: list[Message] = []
        self._phase: _Phase = _Phase.COLLECTING

    def execute(self, user_input: str) -> str:
        """Process one turn of user input and return the AI's next response."""
        if self._phase == _Phase.CONFIRMING:
            return self._handle_confirmation(user_input)

        self._history.append({"role": "user", "content": user_input})

        extracted = self._extract(user_input)
        self._apply(extracted)

        missing = self.validate()
        if not missing:
            if self.auto_confirm:
                self._phase = _Phase.CONFIRMING
                reply = self._generate_confirm_prompt()
            else:
                self._phase = _Phase.COMPLETE
                reply = self.on_complete()
        else:
            reply = self._ask_for(missing[0])

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def on_complete(self) -> str:
        """
        Called when all fields are collected (and confirmed, if auto_confirm).

        Override to customise the completion message or trigger downstream
        effects (e.g. writing to a database, scheduling a follow-up action).
        """
        return "Thank you, I have everything I need."

    def is_complete(self) -> bool:
        """
        True only when all required fields have been collected **and** the
        user has confirmed them (or ``auto_confirm`` is disabled).

        Overrides the base class so that the confirming phase is not
        mistakenly treated as completion.
        """
        return self._phase == _Phase.COMPLETE

    @property
    def history(self) -> list[Message]:
        """Read-only view of the conversation so far."""
        return list(self._history)

    @property
    def phase(self) -> str:
        """Current phase: 'collecting', 'confirming', or 'complete'."""
        return self._phase.name.lower()

    def reset(self) -> None:
        """Clear all field values, conversation history, and phase."""
        super().reset()
        self._history.clear()
        self._phase = _Phase.COLLECTING

    def _generate_confirm_prompt(self) -> str:
        """
        Read back the collected data and ask the user to confirm.

        Uses a deterministic template first so the question is always
        unambiguous ("Is that correct? Yes or no.").  Subclasses can
        override ``confirm_prompt_template`` to customise the phrasing,
        or set ``confirm_prompt_template = ""`` to fall back to an LLM
        call for fully natural language phrasing.
        """
        template = getattr(self.__class__, "confirm_prompt_template", None)

        if template is None:
            summary = self._current_state_description()
            return (
                f"Just to confirm — here's what I have:\n{summary}\n"
                ""
            )

        if template == "":
            system_prompt = "\n\n".join([
                self.system_context,
                "Collected information:\n" + self._current_state_description(),
                (
                    "Read back the collected information to the caller as a natural "
                    "summary, then ask them to confirm with a clear yes-or-no question. "
                    "End your message with 'Is that correct?'"
                ),
            ])
            messages: list[Message] = [
                {"role": "system", "content": system_prompt},
                *self._history,
            ]
            return self._resolve_backend().complete(messages)

        return template.format(**self.collected())

    def _handle_confirmation(self, user_input: str) -> str:
        """
        Process a user response during the confirming phase.

        Classifies the response as 'confirmed' or 'correction'.  On
        correction, transitions back to collecting so the user can amend.
        """
        self._history.append({"role": "user", "content": user_input})
        verdict = self._classify_confirmation(user_input)

        if verdict == "confirmed":
            self._phase = _Phase.COMPLETE
            reply = self.on_complete()
        else:
            self._phase = _Phase.COLLECTING
            extracted = self._extract(user_input)
            self._apply(extracted)
            missing = self.validate()
            if not missing:
                self._phase = _Phase.CONFIRMING
                reply = self._generate_confirm_prompt()
            else:
                reply = self._ask_for(missing[0])

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def _classify_confirmation(self, user_input: str) -> str:
        """
        Ask the LLM whether the user confirmed or wants to make a correction.

        Returns ``"confirmed"`` or ``"correction"``.
        """
        system_prompt = (
            "The user was asked to confirm some collected information. "
            "Classify their response as one of:\n"
            "- confirmed: the user agrees the information is correct\n"
            "- correction: the user wants to change something\n\n"
            "Return ONLY one of these two words, nothing else."
        )
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        result = self._resolve_backend().complete(messages).strip().lower()
        return "confirmed" if "confirmed" in result else "correction"

    def _extract(self, user_input: str) -> dict:
        """
        Ask the LLM to pull field values out of the latest user message.
        Returns a (possibly empty) dict mapping field names to values.
        """
        system_prompt = "\n\n".join([
            "You are a data extraction assistant.",
            "Fields to collect:\n" + self._fields_description(),
            "Already collected:\n" + self._current_state_description(),
            (
                "Extract any field values you can identify from the user's message. "
                "Return ONLY a valid JSON object whose keys are field names and whose "
                "values are the extracted data. Include only fields you are confident about. "
                "If nothing can be extracted, return an empty JSON object: {}"
            ),
        ])
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        response = self._resolve_backend().complete(messages)
        return self._parse_json(response)

    def _ask_for(self, field_name: str) -> str:
        """
        Ask the LLM to continue the conversation and collect the next field.
        Passes full conversation history so the response feels natural.
        """
        field_def = self._fields[field_name]
        description = field_def.description or (
            field_def.label or field_name.replace("_", " ").title()
        )

        system_prompt = "\n\n".join([
            self.system_context,
            "Already collected:\n" + self._current_state_description(),
            "Still needed (required):\n" + self._missing_fields_description(),
            (
                f"Your immediate goal is to collect: {description}\n"
                "Continue the conversation naturally. Ask for exactly one piece of "
                "information. Be brief and friendly — do not list all missing fields."
            ),
        ])
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            *self._history,
        ]
        return self._resolve_backend().complete(messages)

    def _apply(self, extracted: dict) -> None:
        """
        Write extracted values onto the instance *and* the shared Context.

        Context keys are namespaced as ``<namespace>.<field_name>`` to
        avoid collisions with other instructions in the same workflow.
        """
        ns = self.context_namespace or type(self).__name__
        for name, value in extracted.items():
            if name in self._fields and value is not None and value != "":
                object.__setattr__(self, name, value)
                self._context.set(f"{ns}.{name}", value)

    def _resolve_backend(self) -> LLMBackend:
        backend = self._backend or self.__class__.backend
        if backend is None:
            raise RuntimeError(
                f"{type(self).__name__}: no LLM backend configured.\n"
                "Set one at the class level:\n"
                f"    {type(self).__name__}.backend = YourBackend()\n"
                "or pass it to the constructor:\n"
                f"    {type(self).__name__}(backend=YourBackend())"
            )
        return backend

    def _fields_description(self) -> str:
        lines: list[str] = []
        for name, field_def in self._fields.items():
            label = field_def.label or name.replace("_", " ").title()
            req = "required" if field_def.required else "optional"
            desc = field_def.description or label
            lines.append(f"- {name} ({req}): {desc}")
        return "\n".join(lines) if lines else "No fields defined."

    def _current_state_description(self) -> str:
        lines: list[str] = []
        for name, field_def in self._fields.items():
            value = getattr(self, name)
            if value is not None:
                label = field_def.label or name.replace("_", " ").title()
                lines.append(f"- {label}: {value}")
        return "\n".join(lines) if lines else "Nothing collected yet."

    def _missing_fields_description(self) -> str:
        lines: list[str] = []
        for name in self.validate():
            field_def = self._fields[name]
            desc = field_def.description or (
                field_def.label or name.replace("_", " ").title()
            )
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines) if lines else "None."

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if fence:
            text = fence.group(1).strip()
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return {}
