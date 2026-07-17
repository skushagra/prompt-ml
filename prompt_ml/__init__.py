"""
prompt-ml — Prompt Machine Language
====================================
Write LLM prompts as pure Python objects with first-class support for
conditional logic, iteration, recursion, and structured data collection.

Core primitives
---------------
- ``Context``                  — shared data store flowing through every instruction
- ``Field``                    — declare a collectable datum on an instruction
- ``Instruction``              — abstract base for all instruction types

Instruction types
-----------------
- ``DataCollectionInstruction`` — gather structured fields over a conversation
- ``OutputInstruction``         — emit text (static / template / LLM-generated)
- ``ClassifyInstruction``       — map user input to a label (rule-based or LLM)
- ``ActionInstruction``         — execute a Python callable as a side effect

Quick start
-----------
::

    from prompt_ml import Context, Field, DataCollectionInstruction
    from prompt_ml.backend.openai_backend import OpenAIBackend

    ctx = Context()

    class CollectName(DataCollectionInstruction):
        system_context = "You are a friendly phone assistant."
        first_name: str = Field(required=True,  description="Caller's first name")
        last_name:  str = Field(required=False, description="Caller's last name")

    instr = CollectName(backend=OpenAIBackend(api_key="sk-..."), context=ctx)
    reply = instr.execute("Hi, I'm Jane Doe.")
"""

from prompt_ml.context import Context, ContextView
from prompt_ml.fields import Field, FieldDefinition
from prompt_ml.instruction import Instruction
from prompt_ml.instructions.action import ActionInstruction, ActionResult
from prompt_ml.instructions.classify import ClassifyInstruction
from prompt_ml.instructions.data_collection import DataCollectionInstruction
from prompt_ml.instructions.output import OutputInstruction
from prompt_ml.workflow.branch import Branch
from prompt_ml.workflow.flow import Flow
from prompt_ml.workflow.loop import Loop
from prompt_ml.workflow.sequence import Sequence

__all__ = [
    "Context",
    "ContextView",
    "Field",
    "FieldDefinition",
    "Instruction",
    "ActionInstruction",
    "ActionResult",
    "ClassifyInstruction",
    "DataCollectionInstruction",
    "OutputInstruction",
    "Branch",
    "Flow",
    "Loop",
    "Sequence",
]
