"""A statement in a report, and what stands behind it.

Design principle 5 says every claim is cited or explicitly unverified, and this type is
where "explicitly" is made to mean something. An empty list of supporting facts asserts
nothing on its own — it is equally consistent with "nothing was found", "the extraction
failed" and "somebody forgot". So a claim with no facts behind it must carry a reason in
words, and a claim with facts behind it must not: exactly one of the two is always true.

That reason is not the critic's verdict, and the two must not be merged. The reason is the
synthesiser saying it found no support to cite. ``Verdict.NO_SOURCE`` is the critic saying
the support that *was* cited does not bear on the claim. A claim can carry both, and they
would be telling a reader two different things.

``review`` is optional because a claim exists before it is criticised. ``None`` means the
critic has not run on it, which is a different state from any verdict it might return —
including an unfavourable one.
"""

from __future__ import annotations

from typing import Final, Self

from pydantic import Field, model_validator

from researchmind.core.base import DomainModel
from researchmind.core.ids import ClaimId, FactId
from researchmind.core.review import Review

MAX_CLAIM_LENGTH: Final = 1000
"""Longest claim text, in characters."""

MAX_UNVERIFIED_REASON_LENGTH: Final = 500
"""Longest explanation of why a claim carries no citation."""


class Claim(DomainModel):
    """One statement in a report, either cited or explicitly unverified.

    The facts are referenced by identifier rather than embedded, because a single fact
    frequently supports several claims and copying it would let the copies drift.
    Resolving those references is the report's job, where all the facts are in scope.
    """

    id: ClaimId
    text: str = Field(min_length=1, max_length=MAX_CLAIM_LENGTH)
    supported_by: tuple[FactId, ...] = ()
    unverified_reason: str | None = Field(
        default=None, min_length=1, max_length=MAX_UNVERIFIED_REASON_LENGTH
    )
    review: Review | None = None

    @model_validator(mode="after")
    def _require_citation_or_a_stated_absence(self) -> Self:
        """Enforce design principle 5: cited, or unverified in so many words.

        Both at once is incoherent — a claim cannot rest on evidence and report having
        none — and neither is the silent gap the principle exists to forbid.
        """
        if len(set(self.supported_by)) != len(self.supported_by):
            msg = "supported_by cites the same fact more than once"
            raise ValueError(msg)
        if self.supported_by and self.unverified_reason is not None:
            msg = "a claim with supporting facts cannot also be marked unverified"
            raise ValueError(msg)
        if not self.supported_by and self.unverified_reason is None:
            msg = "a claim with no supporting facts must say why it is unverified"
            raise ValueError(msg)
        return self

    @property
    def is_unverified(self) -> bool:
        """Return whether this claim reaches the reader without a citation."""
        return not self.supported_by
