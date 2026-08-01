"""The limits a run is held to, at the three scopes ADR-0004 defines.

This module says what a budget *is*. What a budget *does* — reserving an upper bound
before a call, clamping ``max_tokens`` to what remains, committing the actual cost and
releasing the difference — is the enforcer in phase 4, where there is state to enforce
against. Nothing here decides anything; there is deliberately no ``permits`` and no
``remaining``.

Three dimensions, because the ways a run can run away are genuinely different. Tokens
bound how much context is burned, dollars bound the invoice, and wall-clock bounds a run
that is neither expensive nor large but simply never finishes.

A budget must set at least one of them. A budget that limits nothing is worse than no
budget at all: it produces the paperwork of control without the fact of it.

Wall-clock is a ``timedelta`` rather than a number, for the reason ``Money`` is not a
float. Seconds, milliseconds and token counts are all plain integers, they all look alike,
and adding the wrong pair of them yields a plausible number rather than an error.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import Annotated, Self

from pydantic import AfterValidator, Field, model_validator

from researchmind.core.base import DomainModel
from researchmind.core.money import Money


def _require_positive(amount: Money) -> Money:
    """Reject a limit of zero or less, which forbids rather than bounds."""
    if amount.nanodollars <= 0:
        msg = "a spending limit must be greater than zero"
        raise ValueError(msg)
    return amount


PositiveMoney = Annotated[Money, AfterValidator(_require_positive)]
"""An amount used as a ceiling, which must leave something to spend."""


class BudgetScopeKind(Enum):
    """The three scopes budgets are enforced at (ADR-0004)."""

    CALL = "call"
    SUB_QUESTION = "sub_question"
    RUN = "run"

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``sub_question``."""
        return self.value


class Budget(DomainModel):
    """The ceilings applied at one scope. At least one must be set."""

    scope: BudgetScopeKind
    max_tokens: int | None = Field(default=None, gt=0)
    max_spend: PositiveMoney | None = None
    max_duration: timedelta | None = Field(default=None, gt=timedelta(0))

    @model_validator(mode="after")
    def _require_at_least_one_limit(self) -> Self:
        """Refuse a budget that bounds nothing."""
        if self.max_tokens is None and self.max_spend is None and self.max_duration is None:
            msg = "a budget must set at least one of max_tokens, max_spend or max_duration"
            raise ValueError(msg)
        return self
