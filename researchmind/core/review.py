"""The critic's verdict on a claim, and the quote it is obliged to produce.

ADR-0006 makes the critic a different *task*, not merely a different prompt: it performs a
typed classification rather than writing prose about quality. The four verdicts are that
classification, and they are deliberately not a score — a number would invite averaging,
and "0.6 supported" is not a thing a reader can act on.

The obligation attached to ``SUPPORTED`` is the part that does the work. A critic that may
answer "supported" without producing the span that supports it is a critic that can agree
with anything; requiring the quote is described in ADR-0006 as a stronger anti-hallucination
mechanism than the choice of model. Here it is a constraint on the type, so a review that
approves without evidence cannot be constructed at all.

A review is a separate object rather than a pair of fields on a claim, because "not yet
reviewed" and "reviewed and found wanting" are different states and must not look alike.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Self

from pydantic import Field, model_validator

from researchmind.core.base import DomainModel
from researchmind.core.clock import UtcDatetime
from researchmind.core.ids import SourceId

MAX_SUPPORTING_QUOTE_LENGTH: Final = 2000
"""Longest quote a review may cite, matching the ceiling on a fact's quote."""


class Verdict(Enum):
    """How a claim stands against the sources cited for it (ADR-0006).

    ``NO_SOURCE`` is the critic's finding that the cited evidence does not bear on the
    claim at all. It is not the same statement as a claim the synthesiser marked
    unverified: that one says nothing was found, this one says what was found is beside
    the point.
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    NO_SOURCE = "no_source"

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``partially_supported``."""
        return self.value


class Review(DomainModel):
    """One critic pass over one claim.

    ``supporting_quote`` and ``source_id`` are optional in general and mandatory for a
    verdict of ``SUPPORTED``. The asymmetry is intentional: approving without evidence is
    the failure worth designing out, while a critic that contradicts a claim is welcome to
    quote its reason but is not required to find one.
    """

    verdict: Verdict
    supporting_quote: str | None = Field(
        default=None, min_length=1, max_length=MAX_SUPPORTING_QUOTE_LENGTH
    )
    source_id: SourceId | None = None
    reviewed_at: UtcDatetime

    @model_validator(mode="after")
    def _require_evidence_for_approval(self) -> Self:
        """Refuse a verdict of ``SUPPORTED`` that cites nothing.

        This is ADR-0006's obligation to quote, expressed where it cannot be forgotten.
        """
        if self.verdict is not Verdict.SUPPORTED:
            return self
        if self.supporting_quote is None:
            msg = "a verdict of supported must quote the span that supports the claim"
            raise ValueError(msg)
        if self.source_id is None:
            msg = "a verdict of supported must name the source it quotes"
            raise ValueError(msg)
        return self
