# Prompt Machine Language

Model your prompts and workflows as Python objects.

---

## Installation

```bash
uv add prompt-ml          # or: pip install prompt-ml
uv add openai             # if you want to use OpenAIBackend
```

---

## Core concepts

### `Instruction`

The base unit of work. Subclass it (or one of the concrete types below) to define what an agent step does. Every instruction has:

- `execute(user_input) -> str` — process one turn, return the response
- `is_complete() -> bool` — whether this step is done
- `validate() -> list[str]` — names of required fields not yet filled
- `reset()` — clear all state for reuse
- a `Context` it reads from and writes to

### `Field`

Declare typed, required or optional data attributes on an instruction:

```python
class CollectGuestDetails(DataCollectionInstruction):
    full_name:    str = Field(required=True,  description="Guest full name")
    check_in:     str = Field(required=True,  description="Check-in date (YYYY-MM-DD)")
    check_out:    str = Field(required=True,  description="Check-out date (YYYY-MM-DD)")
    room_type:    str = Field(required=True,  description="Room type: single, double, or suite")
    dietary_needs: str = Field(required=False, description="Any dietary restrictions")
```

The metaclass lifts these out of the class namespace so each instance gets its own mutable slot.

### `Context`

A flat, dotted-key dict shared across all instructions in a workflow. Instructions write their outputs here so downstream steps can read them without any wiring code.

```python
ctx = Context()
ctx["guest.full_name"] = "Aria Patel"
ctx.get("request_type", "general")
ctx.namespace("reservation")["confirmation_no"] = "HTL-4821"
ctx.snapshot()    # plain dict — useful for templates
ctx.as_nested()   # expand dots into nested dicts
```

---

## Instruction types

### `DataCollectionInstruction`

Collects structured fields from a user over multiple turns. Driven by a 3-phase state machine:

1. **collecting** — LLM extracts field values from each user message; asks for missing ones
2. **confirming** — reads collected data back to the user and asks for a yes/no (`auto_confirm = True` by default)
3. **complete** — `is_complete()` returns `True`; collected fields are written to `Context`

```python
from prompt_ml import DataCollectionInstruction, Field
from prompt_ml.backend.openai_backend import OpenAIBackend

class CollectGuestDetails(DataCollectionInstruction):
    context_namespace = "guest"          # keys written as guest.full_name, etc.
    auto_confirm      = True             # default; set False to skip confirm step
    system_context    = "You are a warm and professional hotel concierge at The Grand."

    full_name:     str = Field(required=True,  description="Guest full name")
    check_in:      str = Field(required=True,  description="Check-in date (YYYY-MM-DD)")
    check_out:     str = Field(required=True,  description="Check-out date (YYYY-MM-DD)")
    room_type:     str = Field(required=True,  description="Room type: single, double, or suite")
    dietary_needs: str = Field(required=False, description="Dietary restrictions or preferences")

    def on_complete(self) -> str:
        return (
            f"Perfect, {self.full_name}! I have you down for a {self.room_type} "
            f"from {self.check_in} to {self.check_out}."
        )

instr = CollectGuestDetails(backend=OpenAIBackend(api_key="sk-..."), context=ctx)
reply = instr.execute("Aria Patel, checking in March 3rd, out March 7th, need a suite please")
```

**Key class variables**

| Variable | Default | Effect |
|---|---|---|
| `system_context` | generic assistant | injected into every LLM call as persona / domain context |
| `auto_confirm` | `True` | whether to run the confirming phase before completing |
| `context_namespace` | class name | prefix for context keys written by this instruction |
| `confirm_prompt_template` | `None` (deterministic) | set `""` for LLM-phrased confirm prompt; set a format string for custom phrasing |

---

### `OutputInstruction`

Non-interactive — produces text without waiting for user input. Three modes (priority order):

```python
class Greeting(OutputInstruction):
    text = "Welcome to The Grand. How may I assist you today?"   # 1. static string

class ReservationConfirmed(OutputInstruction):
    template = (
        "Your reservation is confirmed! "
        "Confirmation number: {reservation.confirmation_no}. "
        "We look forward to welcoming you on {guest.check_in}."
    )                                                             # 2. context template

class PersonalisedWelcome(OutputInstruction):
    system_prompt = "Write a warm one-sentence welcome using the guest name if available."
    backend = OpenAIBackend(api_key="sk-...")                    # 3. LLM-generated
```

`auto_advance = True` — a `Sequence` runs this without waiting for user input.

---

### `ClassifyInstruction`

Maps a user message to one of a defined set of labels. Rule-based matching runs first (zero LLM cost); falls back to LLM for anything ambiguous. The result is written to `Context`.

```python
class ClassifyRequest(ClassifyInstruction):
    labels = {
        "reservation": "Make or modify a room reservation",
        "concierge":   "Restaurant, spa, or activity bookings",
        "complaint":   "Report an issue or make a complaint",
        "other":       "Anything else",
    }
    rules = {
        "reservation": ["book", "reserve", "check in", "check-in", "room"],
        "concierge":   ["restaurant", "spa", "tour", "taxi", "activity"],
        "complaint":   ["complaint", "issue", "problem", "broken", "noise"],
    }
    default_label = "other"
    output_key    = "request_type"   # written to ctx["request_type"]
```

`emit_output = False` — the label is returned by `execute()` but not surfaced to the user.

---

### `ActionInstruction`

Wraps a Python callable as a side effect (API call, DB write, etc.). Reads inputs from `Context`, writes results back.

```python
class CreateReservation(ActionInstruction):
    context_namespace = "reservation"
    max_retries       = 1

    def run(self) -> ActionResult:
        name      = self.context.get("guest.full_name")
        check_in  = self.context.get("guest.check_in")
        check_out = self.context.get("guest.check_out")
        room_type = self.context.get("guest.room_type")
        try:
            res = hotel_api.create_reservation(
                name=name, check_in=check_in,
                check_out=check_out, room_type=room_type,
            )
            return ActionResult.ok(
                data={"confirmation_no": res.id, "room_number": res.room},
                message=f"Reservation created. Confirmation: {res.id}.",
            )
        except ApiError as e:
            return ActionResult.fail(message="Could not create reservation.", error=e)

    def on_failure(self, result: ActionResult) -> str:
        return "I'm sorry, I wasn't able to complete your reservation. Please contact the front desk."
```

`auto_advance = True` — runs automatically in a `Sequence` after the preceding step completes.

---

## Workflow primitives

### `Sequence`

Runs steps in order. Auto-drains consecutive `auto_advance` steps without waiting for user input.

```python
Sequence([
    Greeting(),              # auto_advance → fires immediately
    CollectGuestDetails(..), # interactive → waits for guest input
    CreateReservation(..),   # auto_advance → fires when details complete
    ReservationConfirmed(),  # auto_advance → fires immediately after
])
```

### `Branch`

Classifies user input and routes to the matching sub-flow. The original message is also forwarded to the selected route's first step.

```python
Branch(
    classifier=ClassifyRequest(context=ctx, backend=backend),
    routes={
        "reservation": Sequence([CollectGuestDetails(..), CreateReservation(..), ReservationConfirmed()]),
        "concierge":   Sequence([CollectConciergeRequest(..), BookAmenity(..)]),
        "complaint":   LogComplaint(),
        "other":       TransferToFrontDesk(),
    },
    default="other",
)
```

### `Loop`

Repeats a step until a condition is met or `max_iterations` is reached.

```python
Loop(
    step=CollectPaymentDetails(context=ctx, backend=backend),
    until=lambda ctx: ctx.get("payment.verified") is True,
    max_iterations=3,
    context=ctx,
)
```

### `Flow`

Top-level orchestrator. Owns the `Context`, builds the workflow graph in `build()`, and exposes `start()` / `execute()` / `reset()`.

```python
class HotelConciergeFlow(Flow):
    def build(self, context, backend):
        return Sequence([
            Greeting(),
            Branch(
                classifier=ClassifyRequest(context=context, backend=backend),
                routes={
                    "reservation": Sequence([
                        CollectGuestDetails(context=context, backend=backend),
                        CreateReservation(context=context),
                        ReservationConfirmed(context=context),
                    ]),
                    "concierge": ConciergeHandler(context=context, backend=backend),
                    "other":     TransferToFrontDesk(),
                },
                default="other",
            ),
        ])

flow = HotelConciergeFlow(backend=OpenAIBackend(api_key="sk-..."))
print(flow.start())                  # fires auto-advance opening steps

while not flow.is_complete():
    reply = flow.execute(input("Guest: "))
    print("Concierge:", reply)

flow.context.snapshot()              # full collected data
flow.history                         # list of {"user": ..., "agent": ...} dicts
```

---

## Backends

| Backend | Usage |
|---|---|
| `OpenAIBackend(api_key, model, temperature)` | Real OpenAI Chat Completions API |
| `MockBackend(responses=[...])` | Scripted queue for tests; falls back to echo mode when queue is empty |

Backends satisfy the `LLMBackend` protocol — one method: `complete(messages) -> str`. Add any provider by implementing that interface.

---

## Testing

Unit tests use `MockBackend` — no API key needed:

```bash
uv run python -m pytest tests/ -m "not integration" -v
```

Integration tests hit the real OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
uv run python -m pytest tests/ -m integration -v
```
