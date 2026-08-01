"""What one call actually cost, recorded once and never recalculated.

ADR-0004 requires a computed cost to be stored as an absolute number, so that a historical
report does not move when a vendor changes its price list. ``price_version`` is what makes
that rule auditable rather than merely asserted: the number says what was charged and the
version says which price list produced it. Without it the record is absolute but nobody
can check it.

The domain cannot enforce "never recomputed" — nothing stops a caller writing a second
record with a different number. That discipline belongs to storage. What the domain can do
is make the record complete enough that a violation is visible.

``amount`` is non-negative, even though ``Money`` deliberately is not. Both rules are
right. ``Money`` permits negative values because releasing a budget reservation is a
negative movement (ADR-0004), and a cost that has been incurred is not a movement — it is
a quantity, and a negative one would mean a call that paid us. If anyone ever "fixes" the
bound on ``Money``, this is the constraint that was relying on it staying as it is.

``sub_question_id`` is optional because the planner's own calls happen before any
sub-question exists. The correlation chain of design principle 7 — run to step to call —
genuinely has no middle link there, and inventing one would mean inventing a step.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import Annotated, Self

from pydantic import AfterValidator, Field, model_validator

from researchmind.core.base import DomainModel
from researchmind.core.clock import UtcDatetime
from researchmind.core.ids import CallId, RunId, SubQuestionId
from researchmind.core.money import Money
from researchmind.core.tokens import TokenUsage


def _require_non_negative(amount: Money) -> Money:
    """Reject a cost below zero, whatever Money itself permits."""
    if amount.nanodollars < 0:
        msg = "a cost that has been incurred cannot be negative"
        raise ValueError(msg)
    return amount


NonNegativeMoney = Annotated[Money, AfterValidator(_require_non_negative)]
"""An amount that has been spent, as opposed to a movement in either direction."""


class CallKind(Enum):
    """What kind of outbound call incurred a cost.

    The distinction is not cosmetic: it decides whether token counts exist at all.
    """

    LLM = "llm"
    EMBEDDING = "embedding"
    TOOL = "tool"

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``embedding``."""
        return self.value


class Cost(DomainModel):
    """The price of a single call, with everything needed to attribute it.

    Tokens are present exactly when the call had any. A tool call priced per request has
    no token count, and writing zero there would be a measurement of something that was
    never measured.
    """

    call_id: CallId
    run_id: RunId
    sub_question_id: SubQuestionId | None = None
    kind: CallKind
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    tokens: TokenUsage | None = None
    amount: NonNegativeMoney
    latency: timedelta = Field(ge=timedelta(0))
    price_version: str = Field(min_length=1, max_length=64)
    incurred_at: UtcDatetime

    @model_validator(mode="after")
    def _require_tokens_exactly_where_they_exist(self) -> Self:
        """Tie the presence of token counts to the kind of call that was made."""
        if self.kind is CallKind.TOOL and self.tokens is not None:
            msg = "a tool call has no token count; recording one invents a measurement"
            raise ValueError(msg)
        if self.kind is not CallKind.TOOL and self.tokens is None:
            msg = f"a {self.kind} call must report its token usage"
            raise ValueError(msg)
        return self
