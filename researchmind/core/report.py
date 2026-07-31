"""The report: the artefact a reader is handed, and everything needed to check it.

A report carries its plan, its sources, its facts and its claims as objects rather than as
identifiers. That is what the first paragraph of the README promises — a structured
artefact in which every statement can be traced — and it is only deliverable if the trace
travels with the document. A report of identifiers is a report you can check when you also
have the database.

Carrying the objects is also what makes the invariants below checkable at all. Each of them
spans more than one object, so none can be expressed as a field constraint, and none can be
verified by any of the participants alone.

A report has no identifier of its own. A run produces one report, so ``run_id`` names it;
minting a second identity for something that already has one only creates the question of
which report belongs to which run.

Failures here raise :class:`ReferentialIntegrityError`, not a pydantic ``ValidationError``.
The distinction is operational. A malformed payload is bad input and belongs to whoever
sent it; a report assembled from parts that do not fit is a defect in the synthesiser. The
consequence to keep in mind is that this exception travels past ``except ValidationError``,
which is deliberate and is covered by a test.
"""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from researchmind.core.base import DomainModel
from researchmind.core.claim import Claim
from researchmind.core.clock import UtcDatetime
from researchmind.core.errors import CoreError
from researchmind.core.fact import Fact
from researchmind.core.ids import FactId, RunId, SourceId, SubQuestionId
from researchmind.core.plan import Plan
from researchmind.core.question import ResearchQuestion
from researchmind.core.source import Source


class ReferentialIntegrityError(CoreError):
    """Raised when the parts of a report do not refer to each other consistently.

    Every case this covers — a fact citing an absent source, a claim citing an absent
    fact, an identifier used twice — means the assembled document cannot be traced, which
    is the one thing it exists to support.
    """


class Report(DomainModel):
    """A finished report, self-contained enough to be checked without a database.

    ``sources`` and ``facts`` may be empty. A run that searched honestly and found nothing
    still produces a report: one whose claims are every one of them explicitly unverified.
    ``claims`` may not be empty, because a report that states nothing is not an answer.
    """

    run_id: RunId
    question: ResearchQuestion
    plan: Plan
    sources: tuple[Source, ...] = ()
    facts: tuple[Fact, ...] = ()
    claims: tuple[Claim, ...] = Field(min_length=1)
    generated_at: UtcDatetime

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        """Check that every reference in the report resolves inside the report."""
        source_ids = _unique_ids(tuple(source.id for source in self.sources), "source")
        fact_ids = _unique_ids(tuple(fact.id for fact in self.facts), "fact")
        _unique_ids(tuple(claim.id for claim in self.claims), "claim")
        step_ids = frozenset(step.id for step in self.plan.steps)

        _reject_unknown_fact_references(self.facts, source_ids, step_ids)
        _reject_unknown_claim_references(self.claims, fact_ids, source_ids)
        return self


def _unique_ids[IdT: UUID](values: tuple[IdT, ...], kind: str) -> frozenset[IdT]:
    """Return the identifiers as a set, refusing a report that lists one twice.

    A repeated identifier makes every reference to it ambiguous, so this runs before any
    reference is resolved against it.
    """
    unique = frozenset(values)
    if len(unique) != len(values):
        msg = f"the report lists the same {kind} identifier more than once"
        raise ReferentialIntegrityError(msg)
    return unique


def _reject_unknown_fact_references(
    facts: tuple[Fact, ...],
    source_ids: frozenset[SourceId],
    step_ids: frozenset[SubQuestionId],
) -> None:
    """Check that each fact names a source and a plan step the report actually contains.

    The plan step matters as much as the source: a fact attributed to a step that is not
    in the executed revision is a fact nobody can account for, and ADR-0002 exists to make
    that attribution reliable.
    """
    for fact in facts:
        if fact.source_id not in source_ids:
            msg = f"fact {fact.id} cites source {fact.source_id}, which the report omits"
            raise ReferentialIntegrityError(msg)
        if fact.sub_question_id not in step_ids:
            msg = (
                f"fact {fact.id} is attributed to sub-question {fact.sub_question_id}, "
                f"which is not a step of the executed plan"
            )
            raise ReferentialIntegrityError(msg)


def _reject_unknown_claim_references(
    claims: tuple[Claim, ...],
    fact_ids: frozenset[FactId],
    source_ids: frozenset[SourceId],
) -> None:
    """Check that each claim, and each review, cites only what the report contains."""
    for claim in claims:
        for fact_id in claim.supported_by:
            if fact_id not in fact_ids:
                msg = f"claim {claim.id} cites fact {fact_id}, which the report omits"
                raise ReferentialIntegrityError(msg)
        review = claim.review
        if review is None or review.source_id is None:
            continue
        if review.source_id not in source_ids:
            msg = (
                f"the review of claim {claim.id} quotes source {review.source_id}, "
                f"which the report omits"
            )
            raise ReferentialIntegrityError(msg)
