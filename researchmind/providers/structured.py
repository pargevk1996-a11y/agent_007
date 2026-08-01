"""Asking a model for a typed object, and what comes back when it does not produce one.

ADR-0003 gives the provider interface two methods, and this is the shape of the second. The
type parameter is bounded on ``DomainModel``, which is not decoration: a domain model is
closed to unexpected fields, and ``extra="forbid"`` is what produces the
``additionalProperties: false`` that a structured-output schema is required to carry on
every object it describes. The rule phase 2 adopted for its own reasons is the rule this
API enforces.

Both types extend their unstructured counterparts rather than wrapping them. Field bounds,
the alternating-exchange rule and the accounting fields are inherited instead of restated,
and an adapter can build one request from the other. The trade is that a
``StructuredRequest`` handed to plain ``complete`` typechecks and quietly ignores the
schema; that is a real edge and the reason it is written down here.

``value`` is optional, and the invariant on it is deliberately one-directional.

A generation that was cut short or declined cannot have produced a value, so ``value`` must
be absent whenever the call did not end of its own accord. But a call that *did* end
normally may still have produced nothing usable, because the schema sent to the vendor does
not carry every constraint our types impose: string lengths are demoted to prose in the
schema description, so a model can end its turn with a perfectly well-formed object whose
quote is twice as long as ``Fact`` permits. Requiring a value on ``END_TURN`` would make
that outcome unrepresentable — and it is the outcome the retry policy is being built to
consume.

A failed extraction is therefore a result, not an exception. The call was made and billed,
and a caller that cannot see the usage cannot commit it against the budget (ADR-0004). The
raw ``text`` travels on the same object, so the reason is available without a second call.
"""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from researchmind.core.base import DomainModel
from researchmind.providers.completion import CompletionRequest, CompletionResult, StopReason


class StructuredRequest[T: DomainModel](CompletionRequest):
    """One completion that must answer with an instance of ``output_schema``.

    The field is named ``output_schema`` rather than ``schema`` because pydantic reserves
    the shorter name.
    """

    output_schema: type[T]


class StructuredResult[T: DomainModel](CompletionResult):
    """What came back, parsed if it could be parsed."""

    value: T | None = None

    @model_validator(mode="after")
    def _forbid_a_value_the_generation_never_finished(self) -> Self:
        """Check that a value is only claimed for a generation that ran to its own end."""
        if self.value is not None and self.stop_reason is not StopReason.END_TURN:
            msg = f"a generation that ended with {self.stop_reason} cannot have produced a value"
            raise ValueError(msg)
        return self

    @property
    def extracted(self) -> bool:
        """Return whether the call produced a value, which is not the same as succeeding."""
        return self.value is not None
