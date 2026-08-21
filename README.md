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
class CollectAddress(DataCollectionInstruction):
    line1:    str = Field(required=True,  description="Street address")
    city:     str = Field(required=True,  description="City")
    zip_code: str = Field(required=True,  description="5-digit ZIP")
    line2:    str = Field(required=False, description="Apt / suite")
```

The metaclass lifts these out of the class namespace so each instance gets its own mutable slot.

### `Context`

A flat, dotted-key dict shared across all instructions in a workflow. Instructions write their outputs here so downstream steps can read them without any wiring code.

```python
ctx = Context()
ctx["address.city"] = "Springfield"
ctx.get("intent", "other")
ctx.namespace("booking")["booking_id"] = "SVC-001"
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

class CollectAddress(DataCollectionInstruction):
    context_namespace = "address"        # keys written as address.city, etc.
    auto_confirm      = True             # default; set False to skip confirm step
    system_context    = "You are a friendly phone assistant for Acme Auto."

    line1:    str = Field(required=True,  description="Street address line 1")
    city:     str = Field(required=True,  description="City")
    state:    str = Field(required=True,  description="State (two-letter code)")
    zip_code: str = Field(required=True,  description="Five-digit ZIP code")
    line2:    str = Field(required=False, description="Apartment or suite")

    def on_complete(self) -> str:
        return f"Got it — {self.line1}, {self.city} {self.state} {self.zip_code}."

instr = CollectAddress(backend=OpenAIBackend(api_key="sk-..."), context=ctx)
reply = instr.execute("742 Evergreen Terrace, Springfield IL 62701")
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
    text = "Thank you for calling Acme Auto."          # 1. static string

class Confirmation(OutputInstruction):
    template = "Booking {booking.booking_id} confirmed at {booking.location}."  # 2. context template

class PersonalisedGreeting(OutputInstruction):
    system_prompt = "Generate a warm greeting using caller name if available."  # 3. LLM-generated
    backend = OpenAIBackend(api_key="sk-...")
```

`auto_advance = True` — a `Sequence` runs this without waiting for user input.

---

### `ClassifyInstruction`

Maps a user message to one of a defined set of labels. Rule-based matching runs first (zero LLM cost); falls back to LLM for anything ambiguous. The result is written to `Context`.

```python
class ClassifyIntent(ClassifyInstruction):
    labels = {
        "service": "Schedule a service or repair",
        "sales":   "Buy or enquire about a vehicle",
        "other":   "Anything else",
    }
    rules = {
        "service": ["oil change", "brake", "repair", "r:tyre?s?"],  # prefix r: for regex
        "sales":   ["buy", "purchase", "new car"],
    }
    default_label = "other"
    output_key    = "intent"    # written to ctx["intent"]; default is "<ClassName>.label"
```

`emit_output = False` — the label is returned by `execute()` but not surfaced to the user.

---

### `ActionInstruction`

Wraps a Python callable as a side effect (API call, DB write, etc.). Reads inputs from `Context`, writes results back.

```python
class BookService(ActionInstruction):
    context_namespace = "booking"
    max_retries       = 1

    def run(self) -> ActionResult:
        city = self.context.get("address.city")
        try:
            result = booking_api.create(city=city)
            return ActionResult.ok(
                data={"booking_id": result.id},
                message=f"Booked! Reference: {result.id}",
            )
        except ApiError as e:
            return ActionResult.fail(message="Booking failed.", error=e)

    def on_failure(self, result: ActionResult) -> str:
        return "Sorry, I couldn't complete the booking. Please call us directly."
```

`auto_advance = True` — runs automatically in a `Sequence` after the preceding step completes.

---

## Workflow primitives

### `Sequence`

Runs steps in order. Auto-drains consecutive `auto_advance` steps without waiting for user input.

```python
Sequence([
    Greeting(),            # auto_advance → fires immediately
    CollectAddress(...),   # interactive → waits for caller
    BookService(...),      # auto_advance → fires when address complete
    ServiceConfirmation(), # auto_advance → fires immediately after
])
```

### `Branch`

Classifies user input and routes to the matching sub-flow. The original message is also forwarded to the selected route's first step.

```python
Branch(
    classifier=ClassifyIntent(context=ctx, backend=backend),
    routes={
        "service": Sequence([CollectAddress(...), BookService(...), Farewell()]),
        "sales":   SalesFarewell(),
        "other":   GenericFarewell(),
    },
    default="other",
)
```

### `Loop`

Repeats a step until a condition is met or `max_iterations` is reached.

```python
Loop(
    step=CollectAppointment(context=ctx, backend=backend),
    until=lambda ctx: ctx.get("booking.confirmed") is True,
    max_iterations=3,
    context=ctx,
)
```

### `Flow`

Top-level orchestrator. Owns the `Context`, builds the workflow graph in `build()`, and exposes `start()` / `execute()` / `reset()`.

```python
class DealershipFlow(Flow):
    def build(self, context, backend):
        return Sequence([
            Greeting(),
            Branch(
                classifier=ClassifyIntent(context=context, backend=backend),
                routes={"service": Sequence([...]), "sales": SalesFarewell()},
                default="other",
            ),
        ])

flow = DealershipFlow(backend=OpenAIBackend(api_key="sk-..."))
print(flow.start())                 # fires auto-advance opening steps

while not flow.is_complete():
    reply = flow.execute(input("Caller: "))
    print("Agent:", reply)

flow.context.snapshot()             # full collected data
flow.history                        # list of {"user": ..., "agent": ...} dicts
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

Run the mock dealership demo end-to-end:

```bash
uv run python examples/dealership/service_flow.py
```
