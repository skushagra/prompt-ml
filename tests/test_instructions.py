"""
Tests for all prompt-ml instruction types.

All tests use MockBackend — no real LLM credentials required.

The dealership scenario used throughout:
  1.  Greet the caller                  (OutputInstruction — static)
  2.  Classify their intent             (ClassifyInstruction — rules + LLM)
  3.  Collect their address             (DataCollectionInstruction — auto-confirm)
  4.  Book a service appointment        (ActionInstruction)
  5.  Confirm back to the caller        (OutputInstruction — template)
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
from prompt_ml.backend.mock import MockBackend


@pytest.fixture()
def ctx() -> Context:
    return Context()


@pytest.fixture()
def mock() -> MockBackend:
    return MockBackend()


class TestContext:
    def test_basic_get_set(self, ctx):
        ctx["name"] = "Jane"
        assert ctx["name"] == "Jane"
        assert ctx.get("name") == "Jane"
        assert ctx.get("missing", "default") == "default"

    def test_contains(self, ctx):
        ctx["x"] = 1
        assert "x" in ctx
        assert "y" not in ctx

    def test_namespace_write_read(self, ctx):
        view = ctx.namespace("address")
        view["city"] = "Springfield"
        view["state"] = "IL"
        assert ctx["address.city"] == "Springfield"
        assert view["city"] == "Springfield"
        assert view.snapshot() == {"city": "Springfield", "state": "IL"}

    def test_snapshot(self, ctx):
        ctx["a"] = 1
        ctx["b"] = 2
        snap = ctx.snapshot()
        assert snap == {"a": 1, "b": 2}
        snap["c"] = 3
        assert "c" not in ctx

    def test_as_nested(self, ctx):
        ctx["address.city"] = "Springfield"
        ctx["address.state"] = "IL"
        ctx["intent"] = "service"
        nested = ctx.as_nested()
        assert nested == {
            "address": {"city": "Springfield", "state": "IL"},
            "intent": "service",
        }

    def test_update(self, ctx):
        ctx.update({"a": 1, "b": 2})
        assert ctx["a"] == 1
        assert ctx["b"] == 2


class TestOutputInstruction:

    def test_static_mode(self, ctx):
        class Greeting(OutputInstruction):
            text = "Thank you for calling Acme Auto. How can I help you today?"

        reply = Greeting(context=ctx).execute()
        assert reply == "Thank you for calling Acme Auto. How can I help you today?"

    def test_static_ignores_user_input(self, ctx):
        class Greeting(OutputInstruction):
            text = "Hello!"

        assert Greeting(context=ctx).execute("anything") == "Hello!"

    def test_template_mode_dotted_keys(self, ctx):
        ctx["CollectAddress.city"] = "Springfield"
        ctx["CollectAddress.state"] = "IL"

        class ConfirmLocation(OutputInstruction):
            template = "You are calling from {CollectAddress.city}, {CollectAddress.state}."

        reply = ConfirmLocation(context=ctx).execute()
        assert reply == "You are calling from Springfield, IL."

    def test_template_mode_simple_keys(self, ctx):
        ctx["name"] = "Jane"

        class Hello(OutputInstruction):
            template = "Hello, {name}!"

        assert Hello(context=ctx).execute() == "Hello, Jane!"

    def test_template_missing_key_raises(self, ctx):
        class Bad(OutputInstruction):
            template = "Hello, {missing_key}!"

        with pytest.raises(KeyError, match="missing_key"):
            Bad(context=ctx).execute()

    def test_llm_mode(self, ctx):
        mock = MockBackend(responses=["Welcome, valued customer!"])

        class LLMGreeting(OutputInstruction):
            system_prompt = "Generate a warm greeting."

        reply = LLMGreeting(backend=mock, context=ctx).execute()
        assert reply == "Welcome, valued customer!"
        assert mock.call_count == 1

    def test_static_takes_priority_over_template(self, ctx):
        ctx["name"] = "Jane"

        class Both(OutputInstruction):
            text = "Static wins."
            template = "Hello, {name}!"

        assert Both(context=ctx).execute() == "Static wins."

    def test_is_complete_always_true(self, ctx):
        class G(OutputInstruction):
            text = "Hi"

        g = G(context=ctx)
        g.execute()
        assert g.is_complete()


class TestClassifyInstruction:

    class IntentClassifier(ClassifyInstruction):
        labels = {
            "service": "Customer wants to schedule a service or repair",
            "sales":   "Customer wants to buy or enquire about a vehicle",
            "parts":   "Customer needs parts or accessories",
            "other":   "Anything else",
        }
        rules = {
            "service": ["oil change", "tyre", "brake", "repair", "service"],
            "sales":   ["buy", "purchase", "new car", "price", "quote"],
            "parts":   ["parts", "accessory", "accessories"],
        }
        default_label = "other"

    def test_rule_match_service(self, ctx):
        clf = self.IntentClassifier(context=ctx)
        assert clf.execute("I need an oil change") == "service"

    def test_rule_match_sales(self, ctx):
        clf = self.IntentClassifier(context=ctx)
        assert clf.execute("I want to buy a new car") == "sales"

    def test_rule_match_parts(self, ctx):
        clf = self.IntentClassifier(context=ctx)
        assert clf.execute("Do you sell accessories?") == "parts"

    def test_default_label_when_no_match(self, ctx):
        clf = self.IntentClassifier(context=ctx)
        assert clf.execute("Just calling to say hello") == "other"

    def test_writes_label_to_context(self, ctx):
        clf = self.IntentClassifier(context=ctx)
        clf.execute("I need a brake service")
        assert ctx["IntentClassifier.label"] == "service"

    def test_custom_output_key(self, ctx):
        class CustomKey(ClassifyInstruction):
            labels = {"yes": "Yes", "no": "No"}
            rules = {"yes": ["yes", "yep"], "no": ["no", "nope"]}
            output_key = "confirm_result"
            default_label = "no"

        clf = CustomKey(context=ctx)
        clf.execute("yes please")
        assert ctx["confirm_result"] == "yes"

    def test_llm_fallback(self, ctx):
        mock = MockBackend(responses=["service"])

        class LLMClassifier(ClassifyInstruction):
            labels = {"service": "Service", "other": "Other"}
            default_label = "other"

        clf = LLMClassifier(backend=mock, context=ctx)
        result = clf.execute("I have a weird noise from the engine")
        assert result == "service"
        assert mock.call_count == 1

    def test_llm_tolerant_matching(self, ctx):
        mock = MockBackend(responses=["The label is: service"])

        class LLMClassifier(ClassifyInstruction):
            labels = {"service": "Service", "sales": "Sales"}
            default_label = "service"

        clf = LLMClassifier(backend=mock, context=ctx)
        assert clf.execute("something") == "service"

    def test_regex_rule(self, ctx):
        class RegexClassifier(ClassifyInstruction):
            labels = {"sales": "Sales", "other": "Other"}
            rules = {"sales": ["r:buy(ing)?"]}
            default_label = "other"

        clf = RegexClassifier(context=ctx)
        assert clf.execute("I am buying a car") == "sales"
        assert clf.execute("I bought a car") == "other"

    def test_label_and_is_complete(self, ctx):
        clf = self.IntentClassifier(context=ctx)
        assert clf.label is None
        assert not clf.is_complete()
        clf.execute("oil change please")
        assert clf.label == "service"
        assert clf.is_complete()

    def test_reset_clears_label(self, ctx):
        clf = self.IntentClassifier(context=ctx)
        clf.execute("buy a car")
        clf.reset()
        assert clf.label is None
        assert not clf.is_complete()

    def test_reads_context_in_llm_prompt(self, ctx):
        """Context snapshot is included in the LLM system prompt."""
        ctx["caller.tier"] = "VIP"
        mock = MockBackend(responses=["sales"])

        class CTXClassifier(ClassifyInstruction):
            labels = {"sales": "Sales", "other": "Other"}
            default_label = "other"

        CTXClassifier(backend=mock, context=ctx).execute("I need help")
        system_msg = mock.call_log[0][0]["content"]
        assert "VIP" in system_msg


class _AddressInstruction(DataCollectionInstruction):
    context_namespace = "address"
    line1:    str = Field(required=True,  description="Street address line 1")
    line2:    str = Field(required=False, description="Apartment or suite")
    city:     str = Field(required=True,  description="City")
    state:    str = Field(required=True,  description="State (2-letter code)")
    zip_code: str = Field(required=True,  description="ZIP code")


def _address_mock_collecting() -> MockBackend:
    """Scripts a MockBackend for the collecting phase only (auto_confirm=False)."""
    return MockBackend(responses=[
        '{"line1": "742 Evergreen Terrace", "city": "Springfield", "state": "IL"}',
        "Could you give me your ZIP code?",
        '{"zip_code": "62701"}',
    ])


def _address_mock_with_confirm() -> MockBackend:
    """Scripts a MockBackend for the full collecting + confirming flow."""
    return MockBackend(responses=[
        '{"line1": "742 Evergreen Terrace", "city": "Springfield", "state": "IL", "zip_code": "62701"}',
        "I have your address as 742 Evergreen Terrace, Springfield, IL 62701. Is that correct?",
        "confirmed",
    ])


class TestDataCollectionInstruction:

    def test_fields_are_none_on_init(self, ctx):
        class Simple(DataCollectionInstruction):
            name: str = Field(required=True)

        s = Simple(backend=MockBackend(), context=ctx)
        assert s.name is None
        assert not s.is_complete()
        assert s.validate() == ["name"]

    def test_phase_starts_as_collecting(self, ctx):
        instr = _AddressInstruction(backend=MockBackend(), context=ctx)
        assert instr.phase == "collecting"

    def test_extraction_updates_fields(self, ctx):
        mock = MockBackend(responses=[
            '{"line1": "123 Main St", "city": "Portland", "state": "OR"}',
            "What is your ZIP code?",
        ])

        class Addr(DataCollectionInstruction):
            auto_confirm = False
            line1: str = Field(required=True)
            city:  str = Field(required=True)
            state: str = Field(required=True)
            zip_code: str = Field(required=True)

        instr = Addr(backend=mock, context=ctx)
        instr.execute("123 Main St, Portland, Oregon")
        assert instr.line1 == "123 Main St"
        assert instr.city == "Portland"
        assert instr.state == "OR"
        assert instr.zip_code is None

    def test_extracted_fields_written_to_context(self, ctx):
        mock = MockBackend(responses=[
            '{"line1": "742 Evergreen Terrace", "city": "Springfield", "state": "IL"}',
            "What is your ZIP code?",
        ])

        class Addr(DataCollectionInstruction):
            auto_confirm = False
            context_namespace = "address"
            line1: str = Field(required=True)
            city:  str = Field(required=True)
            state: str = Field(required=True)
            zip_code: str = Field(required=True)

        Addr(backend=mock, context=ctx).execute("742 Evergreen Terrace, Springfield IL")
        assert ctx["address.line1"] == "742 Evergreen Terrace"
        assert ctx["address.city"] == "Springfield"
        assert ctx["address.state"] == "IL"
        assert "address.zip_code" not in ctx

    def test_multi_turn_collection_no_confirm(self, ctx):
        """Two turns fill all required fields when auto_confirm=False."""
        class Addr(_AddressInstruction):
            auto_confirm = False

        mock = _address_mock_collecting()
        instr = Addr(backend=mock, context=ctx)

        instr.execute("742 Evergreen Terrace, Springfield, IL")
        assert not instr.is_complete()

        instr.execute("62701")
        assert instr.is_complete()
        assert instr.phase == "complete"

    def test_auto_confirm_transitions_to_confirming(self, ctx):
        mock = _address_mock_with_confirm()
        instr = _AddressInstruction(backend=mock, context=ctx)

        instr.execute("742 Evergreen Terrace, Springfield, IL 62701")
        assert instr.phase == "confirming"
        assert instr.is_complete() is False

    def test_auto_confirm_completes_on_yes(self, ctx):
        mock = _address_mock_with_confirm()
        instr = _AddressInstruction(backend=mock, context=ctx)

        instr.execute("742 Evergreen Terrace, Springfield, IL 62701")
        instr.execute("Yes, that's correct")
        assert instr.phase == "complete"
        assert instr.is_complete()

    def test_auto_confirm_correction_loops_back(self, ctx):
        mock = MockBackend(responses=[
            '{"line1": "742 Evergreen Terrace", "city": "Springfield", "state": "IL", "zip_code": "62701"}',
            "Is 742 Evergreen Terrace, Springfield, IL 62701 correct?",
            "correction",
            '{"zip_code": "62702"}',
            "re-confirm",
            "confirmed",
        ])

        instr = _AddressInstruction(backend=mock, context=ctx)
        instr.execute("742 Evergreen Terrace, Springfield, IL 62701")
        assert instr.phase == "confirming"

        instr.execute("Actually the ZIP is 62702")
        assert instr.phase == "confirming"
        assert instr.zip_code == "62702"

        instr.execute("Yes, confirmed")
        assert instr.phase == "complete"

    def test_history_accumulates(self, ctx):
        mock = MockBackend(responses=[
            '{"line1": "1 Test St"}',
            "What city?",
        ])

        class Simple(DataCollectionInstruction):
            auto_confirm = False
            line1: str = Field(required=True)
            city:  str = Field(required=True)

        instr = Simple(backend=mock, context=ctx)
        instr.execute("1 Test St")
        assert len(instr.history) == 2
        assert instr.history[0]["role"] == "user"
        assert instr.history[1]["role"] == "assistant"

    def test_reset_clears_all_state(self, ctx):
        mock = MockBackend(responses=[
            '{"line1": "742 Evergreen Terrace", "city": "Springfield", "state": "IL", "zip_code": "62701"}',
            "confirm prompt",
            "confirmed",
        ])
        instr = _AddressInstruction(backend=mock, context=ctx)
        instr.execute("full address")
        instr.execute("yes")
        assert instr.phase == "complete"

        instr.reset()
        assert instr.phase == "collecting"
        assert instr.line1 is None
        assert instr.history == []

    def test_optional_field_not_required(self, ctx):
        mock = MockBackend(responses=[
            '{"line1": "1 Test St", "city": "Portland", "state": "OR", "zip_code": "97201"}',
            "confirm?",
            "confirmed",
        ])
        instr = _AddressInstruction(backend=mock, context=ctx)
        instr.execute("1 Test St, Portland, OR 97201")
        instr.execute("yes")
        assert instr.phase == "complete"
        assert instr.line2 is None

    def test_no_backend_raises_clearly(self, ctx):
        class NB(DataCollectionInstruction):
            name: str = Field(required=True)

        with pytest.raises(RuntimeError, match="no LLM backend"):
            NB(context=ctx).execute("hello")

    def test_invalid_json_is_handled_gracefully(self, ctx):
        mock = MockBackend(responses=[
            "not json at all",
            "What is your name?",
        ])

        class Simple(DataCollectionInstruction):
            auto_confirm = False
            name: str = Field(required=True)

        instr = Simple(backend=mock, context=ctx)
        instr.execute("hello")
        assert instr.name is None


class TestActionInstruction:

    def test_success_writes_data_to_context(self, ctx):
        class BookAppointment(ActionInstruction):
            context_namespace = "booking"

            def run(self) -> ActionResult:
                return ActionResult.ok(
                    data={"booking_id": "B-1234", "date": "2024-07-15"},
                    message="Appointment booked!",
                )

        action = BookAppointment(context=ctx)
        reply = action.execute()
        assert reply == "Appointment booked!"
        assert ctx["booking.booking_id"] == "B-1234"
        assert ctx["booking.date"] == "2024-07-15"
        assert action.is_complete()

    def test_failure_re_raises_by_default(self, ctx):
        class AlwaysFails(ActionInstruction):
            def run(self) -> ActionResult:
                return ActionResult.fail(message="API down", error=ConnectionError("down"))

        with pytest.raises(ConnectionError, match="down"):
            AlwaysFails(context=ctx).execute()

    def test_on_failure_can_return_message(self, ctx):
        class GracefulFail(ActionInstruction):
            def run(self) -> ActionResult:
                return ActionResult.fail(message="API down")

            def on_failure(self, result: ActionResult) -> str:
                return "Sorry, something went wrong. Please try again."

        reply = GracefulFail(context=ctx).execute()
        assert reply == "Sorry, something went wrong. Please try again."
        assert not GracefulFail(context=ctx).is_complete()

    def test_retries(self, ctx):
        attempts = []

        class FlakyAction(ActionInstruction):
            max_retries = 2
            context_namespace = "flaky"

            def run(self) -> ActionResult:
                attempts.append(1)
                if len(attempts) < 3:
                    return ActionResult.fail(message="not yet")
                return ActionResult.ok(data={"done": True}, message="ok")

        reply = FlakyAction(context=ctx).execute()
        assert reply == "ok"
        assert len(attempts) == 3
        assert ctx["flaky.done"] is True

    def test_reads_from_context(self, ctx):
        ctx["address.city"] = "Springfield"

        class LookupAction(ActionInstruction):
            context_namespace = "lookup"

            def run(self) -> ActionResult:
                city = self.context.get("address.city")
                return ActionResult.ok(
                    data={"dealership": f"Acme {city}"},
                    message=f"Found dealership in {city}",
                )

        action = LookupAction(context=ctx)
        action.execute()
        assert ctx["lookup.dealership"] == "Acme Springfield"

    def test_result_property(self, ctx):
        class SimpleAction(ActionInstruction):
            def run(self) -> ActionResult:
                return ActionResult.ok(message="done")

        action = SimpleAction(context=ctx)
        assert action.result is None
        action.execute()
        assert action.result is not None
        assert action.result.success

    def test_reset_clears_result(self, ctx):
        class SimpleAction(ActionInstruction):
            def run(self) -> ActionResult:
                return ActionResult.ok(message="done")

        action = SimpleAction(context=ctx)
        action.execute()
        action.reset()
        assert action.result is None
        assert not action.is_complete()


class TestDealershipFlow:
    """
    Simulate a complete service-booking call:

      Greeting  →  Classify intent  →  Collect address
      →  Book appointment  →  Confirm to caller

    All instructions share a single Context.
    """

    def test_full_service_booking_flow(self):
        ctx = Context()

        class Greeting(OutputInstruction):
            text = "Thank you for calling Acme Auto. How can I help you today?"

        greeting = Greeting(context=ctx)
        assert greeting.execute() == "Thank you for calling Acme Auto. How can I help you today?"

        class ClassifyIntent(ClassifyInstruction):
            labels = {
                "service": "Schedule a service or repair",
                "sales":   "Buy or enquire about a vehicle",
                "other":   "Anything else",
            }
            rules = {
                "service": ["oil change", "service", "repair", "brake", "tyre"],
                "sales":   ["buy", "purchase", "new car"],
            }
            default_label = "other"
            output_key = "intent"

        classifier = ClassifyIntent(context=ctx)
        intent = classifier.execute("I need an oil change please")
        assert intent == "service"
        assert ctx["intent"] == "service"

        class CollectAddress(DataCollectionInstruction):
            context_namespace = "address"
            auto_confirm = True
            system_context = "You are a friendly Acme Auto phone assistant."
            line1:    str = Field(required=True,  description="Street address line 1")
            city:     str = Field(required=True,  description="City")
            state:    str = Field(required=True,  description="State")
            zip_code: str = Field(required=True,  description="ZIP code")

        addr_mock = MockBackend(responses=[
            '{"line1": "742 Evergreen Terrace", "city": "Springfield", "state": "IL", "zip_code": "62701"}',
            "I have 742 Evergreen Terrace, Springfield, IL 62701. Is that right?",
            "confirmed",
        ])
        addr = CollectAddress(backend=addr_mock, context=ctx)

        addr.execute("742 Evergreen Terrace Springfield IL 62701")
        assert addr.phase == "confirming"

        addr.execute("Yes, correct")
        assert addr.phase == "complete"

        assert ctx["address.line1"] == "742 Evergreen Terrace"
        assert ctx["address.city"] == "Springfield"
        assert ctx["address.zip_code"] == "62701"

        class BookService(ActionInstruction):
            context_namespace = "booking"
            max_retries = 1

            def run(self) -> ActionResult:
                city = self.context.get("address.city")
                intent = self.context.get("intent")
                if not city or not intent:
                    return ActionResult.fail(message="Missing required context")
                return ActionResult.ok(
                    data={"booking_id": "SVC-001", "location": city},
                    message=f"Booked a {intent} appointment in {city}!",
                )

        booking = BookService(context=ctx)
        booking_reply = booking.execute()
        assert booking_reply == "Booked a service appointment in Springfield!"
        assert ctx["booking.booking_id"] == "SVC-001"
        assert booking.is_complete()

        class BookingConfirmation(OutputInstruction):
            template = (
                "Perfect! Your service appointment (ref: {booking.booking_id}) "
                "has been scheduled for our {booking.location} location. "
                "We'll send a confirmation to the address on file. Goodbye!"
            )

        confirmation = BookingConfirmation(context=ctx)
        text = confirmation.execute()
        assert "SVC-001" in text
        assert "Springfield" in text
        assert text == (
            "Perfect! Your service appointment (ref: SVC-001) "
            "has been scheduled for our Springfield location. "
            "We'll send a confirmation to the address on file. Goodbye!"
        )

        snap = ctx.snapshot()
        assert snap["intent"] == "service"
        assert snap["address.city"] == "Springfield"
        assert snap["booking.booking_id"] == "SVC-001"
