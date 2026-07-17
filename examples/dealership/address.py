"""
Example: CollectAddress instruction for an auto dealership phone AI.

This file shows how to define a DataCollectionInstruction in its entirety:
just declare Field() attributes.  No execute(), no validate(), no prompting
logic — all of that is inherited.

Running this file directly runs a simulated multi-turn conversation using
the MockBackend so no real LLM credentials are needed.
"""

from __future__ import annotations

from prompt_ml.backend.mock import MockBackend
from prompt_ml.fields import Field
from prompt_ml.instructions.data_collection import DataCollectionInstruction


# ---------------------------------------------------------------------------
# Shared dealership persona (can be set on a base class and inherited across
# all dealership-specific instructions).
# ---------------------------------------------------------------------------

class DealershipInstruction(DataCollectionInstruction):
    """Base class that injects the dealership phone-AI persona."""

    system_context = (
        "You are a friendly phone assistant for Acme Auto Dealership. "
        "Keep responses concise — the caller is on the phone. "
        "Speak naturally and avoid technical jargon."
    )


# ---------------------------------------------------------------------------
# The actual instruction: just fields.  Everything else is inherited.
# ---------------------------------------------------------------------------

class CollectAddress(DealershipInstruction):
    """Collect a caller's full mailing address over a phone conversation."""

    line1:    str = Field(required=True,  description="Street address line 1")
    line2:    str = Field(required=False, description="Apartment, suite, or unit number")
    city:     str = Field(required=True,  description="City name")
    state:    str = Field(required=True,  description="State (two-letter abbreviation, e.g. IL)")
    zip_code: str = Field(required=True,  description="Five-digit ZIP code")

    def on_complete(self) -> str:
        addr = self.line1
        if self.line2:
            addr += f", {self.line2}"
        addr += f", {self.city}, {self.state} {self.zip_code}"
        return (
            f"Perfect, I have your address as: {addr}. "
            "Is that correct?"
        )


# ---------------------------------------------------------------------------
# Simulated conversation — runs when this file is executed directly.
# ---------------------------------------------------------------------------

def simulate() -> None:
    """
    Walk through a realistic multi-turn address-collection conversation using
    scripted MockBackend responses so no real LLM is required.

    The mock simulates:
      Turn 1: user gives most of the address in one shot
      Turn 2: user supplies the missing ZIP code
    """

    # Script the extraction responses the "LLM" will return.
    # In production these come from a real model.
    mock = MockBackend(responses=[
        # Turn 1 — extraction: model pulls out everything except zip_code
        '{"line1": "742 Evergreen Terrace", "city": "Springfield", "state": "IL"}',
        # Turn 1 — ask_for: model generates a question for zip_code
        "Could you give me your ZIP code?",
        # Turn 2 — extraction: model pulls out zip_code from user reply
        '{"zip_code": "62701"}',
        # Turn 2 — on_complete is called, no extra LLM call needed
    ])

    instr = CollectAddress(backend=mock)

    # ---- Turn 1 -----------------------------------------------------------
    user_says = "Sure, I live at 742 Evergreen Terrace, Springfield, Illinois."
    print(f"\nUser:  {user_says}")
    reply = instr.execute(user_says)
    print(f"Agent: {reply}")
    print(f"       [state: {instr.collected()}]")

    # ---- Turn 2 -----------------------------------------------------------
    user_says = "The ZIP is 62701."
    print(f"\nUser:  {user_says}")
    reply = instr.execute(user_says)
    print(f"Agent: {reply}")
    print(f"       [complete: {instr.is_complete()}]")
    print(f"       [state: {instr.collected()}]")


if __name__ == "__main__":
    simulate()
