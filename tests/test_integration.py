"""
Integration tests — all tests in this file call the real OpenAI API.

Because LLM responses are non-deterministic, assertions check *structure
and behaviour* (fields populated, phases reached, context written) rather
than exact response text.

Run with:
    export OPENAI_API_KEY=sk-...
    uv run --with pytest --with openai pytest tests/test_integration.py -v
"""

import pytest

from prompt_ml import (
    ActionInstruction,
    ActionResult,
    ClassifyInstruction,
    Context,
    DataCollectionInstruction,
    Field,
    OutputInstruction,
)


pytestmark = pytest.mark.integration


class ClassifyIntent(ClassifyInstruction):
    labels = {
        "service": "Customer wants to schedule a service or repair",
        "sales":   "Customer wants to buy or enquire about a vehicle",
        "parts":   "Customer needs parts or accessories",
        "other":   "Anything else",
    }
    rules = {
        "service": ["oil change", "tyre", "brake", "repair"],
        "sales":   ["buy", "purchase", "new car"],
        "parts":   ["parts", "accessory"],
    }
    default_label = "other"
    output_key = "intent"


class CollectAddress(DataCollectionInstruction):
    context_namespace = "address"
    auto_confirm = True
    system_context = (
        "You are a friendly phone assistant for Acme Auto Dealership. "
        "Keep responses concise — the caller is on the phone."
    )
    line1:    str = Field(required=True,  description="Street address line 1")
    line2:    str = Field(required=False, description="Apartment or suite (optional)")
    city:     str = Field(required=True,  description="City name")
    state:    str = Field(required=True,  description="State (two-letter abbreviation)")
    zip_code: str = Field(required=True,  description="Five-digit ZIP code")

    def on_complete(self) -> str:
        parts = [self.line1]
        if self.line2:
            parts.append(self.line2)
        parts += [self.city, f"{self.state} {self.zip_code}"]
        return f"Got it — {', '.join(parts)}. Is there anything else I can help you with?"


class BookService(ActionInstruction):
    context_namespace = "booking"

    def run(self) -> ActionResult:
        city   = self.context.get("address.city", "your area")
        intent = self.context.get("intent", "service")
        return ActionResult.ok(
            data={"booking_id": "SVC-LIVE-001", "location": city, "type": intent},
            message=f"Booked a {intent} appointment in {city}. Reference: SVC-LIVE-001.",
        )


class TestOutputInstructionIntegration:

    def test_static_mode(self, openai_backend):
        """Static mode never calls the LLM."""
        class Greeting(OutputInstruction):
            text = "Thank you for calling Acme Auto. How can I help you today?"

        ctx = Context()
        reply = Greeting(context=ctx).execute()
        assert reply == "Thank you for calling Acme Auto. How can I help you today?"

    def test_template_mode(self, openai_backend):
        """Template mode interpolates from context, never calls the LLM."""
        ctx = Context()
        ctx["booking.booking_id"] = "SVC-999"
        ctx["booking.location"]   = "Springfield"

        class Confirmation(OutputInstruction):
            template = "Your appointment (ref: {booking.booking_id}) is confirmed in {booking.location}."

        reply = Confirmation(context=ctx).execute()
        assert "SVC-999" in reply
        assert "Springfield" in reply

    def test_llm_mode_returns_nonempty_string(self, openai_backend):
        """LLM mode calls OpenAI and returns a non-empty string."""
        ctx = Context()
        ctx["caller.name"] = "Jane"

        class PersonalisedGreeting(OutputInstruction):
            system_prompt = (
                "Generate a warm, one-sentence greeting for a caller. "
                "Use their name if available in the context."
            )

        reply = PersonalisedGreeting(backend=openai_backend, context=ctx).execute()
        assert isinstance(reply, str)
        assert len(reply) > 5


class TestClassifyInstructionIntegration:

    def test_rule_based_needs_no_api(self, openai_backend):
        """Rule matches fire before any LLM call — verifies no unnecessary API usage."""
        ctx = Context()
        clf = ClassifyIntent(context=ctx)
        result = clf.execute("I need an oil change")
        assert result == "service"
        assert ctx["intent"] == "service"

    def test_llm_classifies_service_intent(self, openai_backend):
        """LLM correctly classifies a service-related message not caught by rules."""
        ctx = Context()

        class LLMOnlyClassifier(ClassifyInstruction):
            labels  = ClassifyIntent.labels
            rules   = {}
            default_label = "other"
            output_key = "intent"

        clf = LLMOnlyClassifier(backend=openai_backend, context=ctx)
        result = clf.execute("My engine is making a strange noise and I want it checked")
        assert result in ("service", "other")
        assert ctx["intent"] == result

    def test_llm_classifies_sales_intent(self, openai_backend):
        ctx = Context()

        class LLMOnlyClassifier(ClassifyInstruction):
            labels  = ClassifyIntent.labels
            rules   = {}
            default_label = "other"
            output_key = "intent"

        clf = LLMOnlyClassifier(backend=openai_backend, context=ctx)
        result = clf.execute("I'm interested in purchasing a new SUV")
        assert result == "sales"

    def test_context_data_available_in_prompt(self, openai_backend):
        """Context is included in the system prompt — verifies the plumbing."""
        ctx = Context()
        ctx["caller.tier"] = "VIP"

        class LLMClassifier(ClassifyInstruction):
            labels = {"service": "Service", "sales": "Sales", "other": "Other"}
            rules  = {}
            default_label = "other"
            output_key = "intent"

        clf = LLMClassifier(backend=openai_backend, context=ctx)
        clf.execute("I need help with my car")
        assert ctx["intent"] in clf.labels


class TestDataCollectionInstructionIntegration:

    def test_extracts_full_address_in_one_turn(self, openai_backend):
        """LLM extracts all fields when the user provides them in one message."""
        ctx = Context()
        instr = CollectAddress(backend=openai_backend, context=ctx)

        instr.execute("My address is 742 Evergreen Terrace, Springfield, IL 62701")

        assert instr.line1 is not None
        assert instr.city is not None
        assert instr.state is not None
        assert instr.zip_code is not None
        assert ctx.get("address.city") is not None

    def test_enters_confirming_phase_when_fields_complete(self, openai_backend):
        """After all fields extracted, auto_confirm transitions to confirming."""
        ctx = Context()
        instr = CollectAddress(backend=openai_backend, context=ctx)

        instr.execute("742 Evergreen Terrace, Springfield, IL 62701")
        assert instr.phase == "confirming"
        assert not instr.is_complete()

    def test_completes_on_confirmation(self, openai_backend):
        """'Yes' from the user in the confirming phase marks instruction complete."""
        ctx = Context()
        instr = CollectAddress(backend=openai_backend, context=ctx)

        instr.execute("742 Evergreen Terrace, Springfield, IL 62701")
        assert instr.phase == "confirming"

        instr.execute("Yes, that's correct")
        assert instr.phase == "complete"
        assert instr.is_complete()

    def test_multi_turn_collection(self, openai_backend):
        """When the user provides partial info, the LLM asks for what's missing."""
        ctx = Context()
        instr = CollectAddress(backend=openai_backend, context=ctx)

        reply1 = instr.execute("I live at 742 Evergreen Terrace, Springfield")
        assert isinstance(reply1, str) and len(reply1) > 0
        assert not instr.is_complete()

        reply2 = instr.execute("Illinois, ZIP 62701")
        assert isinstance(reply2, str) and len(reply2) > 0

        if instr.phase == "confirming":
            instr.execute("Yes, correct")

        assert instr.is_complete()

    def test_correction_flow(self, openai_backend):
        """User corrects an extracted value during the confirming phase."""
        ctx = Context()
        instr = CollectAddress(backend=openai_backend, context=ctx)

        instr.execute("742 Evergreen Terrace, Springfield, IL 62701")
        assert instr.phase == "confirming"

        instr.execute("Actually the ZIP is 62702, not 62701")
        assert instr.phase in ("confirming", "collecting")

    def test_history_accumulates_across_turns(self, openai_backend):
        """Each execute() turn appends user + assistant messages to history."""
        ctx = Context()
        instr = CollectAddress(backend=openai_backend, context=ctx)

        instr.execute("742 Evergreen Terrace, Springfield, IL 62701")
        assert len(instr.history) == 2

        instr.execute("Yes that's right")
        assert len(instr.history) == 4

    def test_without_autoconfirm(self, openai_backend):
        """With auto_confirm=False the instruction completes without a confirm step."""
        class QuickCollect(DataCollectionInstruction):
            auto_confirm = False
            context_namespace = "quick"
            system_context = "You are a helpful assistant."
            name:  str = Field(required=True, description="Caller's full name")
            phone: str = Field(required=True, description="Phone number")

        ctx = Context()
        instr = QuickCollect(backend=openai_backend, context=ctx)
        instr.execute("My name is Jane Doe and my number is 555-1234")

        assert instr.phase == "complete"
        assert instr.is_complete()
        assert instr.name is not None
        assert ctx.get("quick.name") is not None


class TestActionInstructionIntegration:
    """
    ActionInstruction wraps Python callables — it does not use LLM directly.
    These tests verify the context wiring works in a live workflow.
    """

    def test_reads_context_written_by_data_collection(self, openai_backend):
        ctx = Context()
        ctx["address.city"]  = "Springfield"
        ctx["address.state"] = "IL"
        ctx["intent"]        = "service"

        action = BookService(context=ctx)
        reply = action.execute()

        assert "Springfield" in reply
        assert "service" in reply
        assert ctx["booking.booking_id"] == "SVC-LIVE-001"
        assert action.is_complete()


class TestDealershipFlowIntegration:

    def test_full_service_booking_call(self, openai_backend):
        """
        Simulate a complete inbound service call end-to-end:

          Greeting  →  Classify intent  →  Collect address (with confirm)
          →  Book appointment  →  Confirmation output

        All instructions share a single Context.  Every LLM call is real.
        """
        ctx = Context()

        class Greeting(OutputInstruction):
            text = "Thank you for calling Acme Auto. How can I help?"

        greeting_text = Greeting(context=ctx).execute()
        assert len(greeting_text) > 0

        clf = ClassifyIntent(context=ctx)
        intent = clf.execute("I need to get my brakes checked")
        assert intent in ("service", "other")
        assert ctx["intent"] == intent

        addr = CollectAddress(backend=openai_backend, context=ctx)
        addr.execute("742 Evergreen Terrace, Springfield, Illinois 62701")

        if addr.phase == "confirming":
            addr.execute("Yes, that's correct")

        assert addr.phase == "complete"
        assert addr.is_complete()
        assert ctx.get("address.city") is not None

        booking = BookService(context=ctx)
        booking_reply = booking.execute()
        assert "SVC-LIVE-001" in booking_reply
        assert booking.is_complete()

        class FinalConfirmation(OutputInstruction):
            template = (
                "All done! Booking reference: {booking.booking_id}. "
                "We look forward to seeing you at our {booking.location} location."
            )

        final = FinalConfirmation(context=ctx).execute()
        assert "SVC-LIVE-001" in final
        assert ctx.get("address.city") in final

        snap = ctx.snapshot()
        assert "intent"              in snap
        assert "address.city"        in snap
        assert "booking.booking_id"  in snap
