from __future__ import annotations

import re
from typing import ClassVar

from prompt_ml.backend.base import LLMBackend, Message
from prompt_ml.context import Context
from prompt_ml.instruction import Instruction


class ClassifyInstruction(Instruction):
    """
    A single-turn instruction that categorises user input into one of a
    defined set of labels, then writes the result to the shared Context.

    Because data is first-class, the classifier has read access to
    everything collected so far via ``self.context``.  Both the rule-based
    and LLM classification paths may use context data to make smarter
    decisions (e.g. classify differently based on a customer tier that was
    looked up earlier).

    Defining labels
    ---------------
    Declare ``labels`` as a class-level dict mapping label names to plain
    English descriptions.  Descriptions are used in the LLM prompt::

        class ClassifyIntent(ClassifyInstruction):
            labels = {
                "service":  "Customer wants to schedule a service or repair",
                "sales":    "Customer wants to buy or enquire about a vehicle",
                "parts":    "Customer needs parts or accessories",
                "other":    "Anything else",
            }

    Rule-based matching (optional)
    --------------------------------
    Declare ``rules`` as a dict mapping label names to lists of keywords or
    regex patterns.  Rules are evaluated **before** the LLM, making them a
    cheap first pass::

        class ClassifyIntent(ClassifyInstruction):
            rules = {
                "service": ["oil change", "tyre", "brake", "repair", "service"],
                "sales":   ["buy", "purchase", "new car", "price", "quote"],
            }

    Rules use case-insensitive substring matching by default.  Prefix a
    pattern with ``r:`` to treat it as a regular expression::

        rules = {"sales": ["r:buy(ing)?", "purchase"]}

    Classification priority
    -----------------------
    1. ``rules`` — if any keyword/pattern matches, return that label
       immediately (no LLM call).
    2. LLM — if ``backend`` is set, ask the model to pick from ``labels``.
    3. ``default_label`` — if set and no other method matched, return it.
    4. ``RuntimeError`` — if nothing matched and no default is set.

    Context output
    --------------
    The resulting label is written to the shared Context under
    ``output_key`` (default: ``"<ClassName>.label"``).  Downstream
    instructions and workflow branches can read this key to route
    execution::

        ctx["ClassifyIntent.label"]   # → "service"
    """

    labels: ClassVar[dict[str, str]] = {}

    rules: ClassVar[dict[str, list[str]]] = {}

    backend: ClassVar[LLMBackend | None] = None

    output_key: ClassVar[str] = ""

    default_label: ClassVar[str] = ""

    emit_output: ClassVar[bool] = False

    def __init__(
        self,
        backend: LLMBackend | None = None,
        context: Context | None = None,
    ) -> None:
        super().__init__(context=context)
        self._backend: LLMBackend | None = backend
        self._label: str | None = None

    def execute(self, user_input: str) -> str:
        """
        Classify ``user_input`` and return the resulting label.

        The label is also written to the shared Context so downstream
        instructions can branch on it.
        """
        if not self.labels:
            raise RuntimeError(
                f"{type(self).__name__}: `labels` dict must be defined."
            )

        label = self._classify(user_input)
        self._label = label

        key = self.output_key or f"{type(self).__name__}.label"
        self._context.set(key, label)

        return label

    @property
    def label(self) -> str | None:
        """The last classification result, or None if not yet executed."""
        return self._label

    def is_complete(self) -> bool:
        return self._label is not None

    def reset(self) -> None:
        super().reset()
        self._label = None

    def _classify(self, user_input: str) -> str:
        rule_match = self._match_rules(user_input)
        if rule_match:
            return rule_match

        backend = self._backend or self.__class__.backend
        if backend:
            return self._llm_classify(user_input, backend)

        if self.default_label:
            return self.default_label

        raise RuntimeError(
            f"{type(self).__name__}: could not classify input and no "
            "`default_label` is set.  Either add matching rules, configure "
            "a backend, or set `default_label`."
        )

    def _match_rules(self, user_input: str) -> str | None:
        """
        Evaluate ``rules`` against ``user_input``.

        Patterns prefixed with ``r:`` are compiled as regexes; all others
        use case-insensitive substring matching.

        The classifier can also inspect ``self.context`` here to apply
        data-aware rules before falling back to the LLM.
        """
        lowered = user_input.lower()
        for label, patterns in self.rules.items():
            if label not in self.labels:
                continue
            for pattern in patterns:
                if pattern.startswith("r:"):
                    if re.search(pattern[2:], user_input, re.IGNORECASE):
                        return label
                else:
                    if pattern.lower() in lowered:
                        return label
        return None

    def _llm_classify(self, user_input: str, backend: LLMBackend) -> str:
        """
        Ask the LLM to classify the input and return a label from ``labels``.

        The full shared Context snapshot is included so the model can make
        data-aware decisions (e.g. treat VIP customers differently).
        """
        label_list = "\n".join(
            f"- {label}: {desc}" for label, desc in self.labels.items()
        )

        context_summary = "\n".join(
            f"  {k}: {v}" for k, v in sorted(self._context.snapshot().items())
        ) or "  (empty)"

        system_prompt = (
            "Classify the user's message into exactly one of the following labels.\n\n"
            f"Labels:\n{label_list}\n\n"
            f"Additional context collected so far:\n{context_summary}\n\n"
            "Return ONLY the label name — nothing else, no explanation."
        )
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        raw = backend.complete(messages).strip().lower()

        for label in self.labels:
            if label.lower() in raw:
                return label

        return self.default_label or next(iter(self.labels))
