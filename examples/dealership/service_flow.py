"""
Full dealership service-booking flow — demonstrates every workflow primitive.

Structure
---------
    Sequence [
        Greeting          (OutputInstruction  — auto_advance)
        Branch [
            classifier:   ClassifyIntent      (ClassifyInstruction)
            "service" →   Sequence [
                              CollectAddress  (DataCollectionInstruction, auto_confirm)
                              BookService     (ActionInstruction         — auto_advance)
                              ServiceConfirmation (OutputInstruction     — auto_advance)
                          ]
            "sales"   →   SalesFarewell       (OutputInstruction        — auto_advance)
            "other"   →   GenericFarewell     (OutputInstruction        — auto_advance)
        ]
    ]

Run with MockBackend (no API key needed):
    uv run python examples/dealership/service_flow.py
"""

from __future__ import annotations

from prompt_ml import (
    ActionInstruction,
    ActionResult,
    Branch,
    ClassifyInstruction,
    Context,
    DataCollectionInstruction,
    Field,
    Flow,
    OutputInstruction,
    Sequence,
)
from prompt_ml.backend.base import LLMBackend
from prompt_ml.backend.mock import MockBackend


class Greeting(OutputInstruction):
    text = "Thank you for calling Acme Auto Dealership. How can I help you today?"


class ClassifyIntent(ClassifyInstruction):
    labels = {
        "service": "Schedule a service, repair, or maintenance appointment",
        "sales":   "Buy a vehicle or get a price quote",
        "other":   "Anything else",
    }
    rules = {
        "service": ["oil change", "service", "repair", "brake", "tyre", "maintenance"],
        "sales":   ["buy", "purchase", "new car", "price", "quote"],
    }
    default_label = "other"
    output_key = "intent"


class CollectAddress(DataCollectionInstruction):
    context_namespace = "address"
    auto_confirm      = True
    system_context    = (
        "You are a friendly phone assistant for Acme Auto Dealership. "
        "Keep responses concise — the caller is on the phone."
    )
    line1:    str = Field(required=True,  description="Street address line 1")
    line2:    str = Field(required=False, description="Apartment or suite (optional)")
    city:     str = Field(required=True,  description="City name")
    state:    str = Field(required=True,  description="State (two-letter code)")
    zip_code: str = Field(required=True,  description="Five-digit ZIP code")

    def on_complete(self) -> str:
        parts = [self.line1]
        if self.line2:
            parts.append(self.line2)
        parts += [self.city, f"{self.state} {self.zip_code}"]
        return f"Got it, I have your address as {', '.join(parts)}."


class BookService(ActionInstruction):
    context_namespace = "booking"

    def run(self) -> ActionResult:
        city = self.context.get("address.city", "your area")
        return ActionResult.ok(
            data={"booking_id": "SVC-001", "location": city},
            message=f"Your service appointment has been booked in {city}.",
        )


class ServiceConfirmation(OutputInstruction):
    template = (
        "All set! Your booking reference is {booking.booking_id}. "
        "We look forward to seeing you at our {booking.location} location. Goodbye!"
    )


class SalesFarewell(OutputInstruction):
    text = "I'll transfer you to our sales team now. Please hold — they'll be right with you!"


class GenericFarewell(OutputInstruction):
    text = "Thanks for calling Acme Auto. Have a great day!"


class DealershipFlow(Flow):
    """
    Complete inbound call handler for Acme Auto Dealership.

    Pass a real backend for production use, or a MockBackend for testing.
    """

    def build(self, context: Context, backend: LLMBackend | None):
        service_branch = Sequence([
            CollectAddress(context=context, backend=backend),
            BookService(context=context),
            ServiceConfirmation(context=context),
        ])

        return Sequence([
            Greeting(),
            Branch(
                classifier=ClassifyIntent(context=context, backend=backend),
                routes={
                    "service": service_branch,
                    "sales":   SalesFarewell(),
                    "other":   GenericFarewell(),
                },
                default="other",
            ),
        ])


def _make_service_mock() -> MockBackend:
    """Scripts a full service-booking call across three caller turns.

    Turn 1: "I need an oil change"
      → ClassifyIntent classifies as "service"
      → CollectAddress receives same message, extracts nothing (no address)
      → CollectAddress asks for street address

    Turn 2: "742 Evergreen Terrace, Springfield, Illinois 62701"
      → CollectAddress extracts all fields
      → enters confirming phase, generates confirm prompt

    Turn 3: "Yes, that's correct"
      → confirmation classified as "confirmed"
      → BookService fires, ServiceConfirmation fires
    """
    return MockBackend(responses=[
        "{}",
        "Sure! Could I get your street address?",
        '{"line1": "742 Evergreen Terrace", "city": "Springfield", "state": "IL", "zip_code": "62701"}',
        "confirmed",
    ])


def _make_sales_mock() -> MockBackend:
    """Scripts a sales enquiry that transfers the caller."""
    return MockBackend(responses=[
        "sales",
    ])


def run_simulation(label: str, mock: MockBackend, turns: list[str]) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Scenario: {label}")
    print(f"{'═' * 60}")

    flow = DealershipFlow(backend=mock)

    opening = flow.start()
    print(f"\nAgent: {opening}")

    for user_says in turns:
        print(f"Caller: {user_says}")
        reply = flow.execute(user_says)
        print(f"Agent:  {reply}")
        if flow.is_complete():
            break

    print(f"\n[Flow complete: {flow.is_complete()}]")
    print(f"[Context snapshot: {flow.context.snapshot()}]")


if __name__ == "__main__":
    run_simulation(
        label="Service appointment",
        mock=_make_service_mock(),
        turns=[
            "I need an oil change",
            "742 Evergreen Terrace, Springfield, Illinois 62701",
            "Yes, that's correct",
        ],
    )

    run_simulation(
        label="Sales enquiry",
        mock=_make_sales_mock(),
        turns=[
            "I'd like to buy a new car",
        ],
    )
