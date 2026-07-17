from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type


@dataclass
class FieldDefinition:
    """
    Metadata attached to a single collectable datum on an Instruction.

    Instances of this class are placed as class-body attributes on an
    Instruction subclass.  The metaclass lifts them out of the class
    namespace and stores them in `_fields`, so that at runtime each
    instance carries its *own* value for every field (not a shared
    class-level constant).
    """

    type: Type = str
    required: bool = True
    description: str = ""
    default: Any = None
    label: str = ""

    def __repr__(self) -> str:
        req = "required" if self.required else "optional"
        return (
            f"FieldDefinition(type={self.type.__name__}, {req}, "
            f"description={self.description!r})"
        )


def Field(
    *,
    required: bool = True,
    type: Type = str,
    description: str = "",
    default: Any = None,
    label: str = "",
) -> FieldDefinition:
    """
    Declare a collectable field on a DataCollectionInstruction.

    Usage::

        class CollectAddress(DataCollectionInstruction):
            line1:    str = Field(required=True,  description="Street address line 1")
            line2:    str = Field(required=False, description="Apartment or suite")
            city:     str = Field(required=True,  description="City")
            state:    str = Field(required=True,  description="State (2-letter code)")
            zip_code: str = Field(required=True,  description="5-digit ZIP code")
    """
    return FieldDefinition(
        type=type,
        required=required,
        description=description,
        default=default,
        label=label,
    )
