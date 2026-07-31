"""Report assembly checked over generated documents.

The interesting claim is not that one hand-built report validates, but that the
referential invariants hold for any shape of report and fail for every way of breaking
one. Identifiers that participate in an invariant are drawn so that a counterexample can
be replayed; ``run_id`` and the plan's own identifier take part in nothing and are minted.
"""

from hashlib import sha256

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from researchmind.core.claim import Claim
from researchmind.core.clock import utc_now
from researchmind.core.confidence import Confidence
from researchmind.core.fact import Fact
from researchmind.core.ids import (
    ClaimId,
    FactId,
    SourceId,
    SubQuestionId,
    new_plan_id,
    new_run_id,
)
from researchmind.core.plan import Plan, PlanAuthor, PlanStatus, SubQuestion
from researchmind.core.question import ResearchQuestion
from researchmind.core.report import ReferentialIntegrityError, Report
from researchmind.core.source import Source

QUESTION = ResearchQuestion(text="A generated research question?")
DIGEST = sha256(b"generated document").hexdigest()

CLAIM_IDS = st.uuids().map(ClaimId)
FACT_IDS = st.uuids().map(FactId)
SOURCE_IDS = st.uuids().map(SourceId)
STEP_IDS = st.uuids().map(SubQuestionId)
REASONS = st.text(min_size=1, max_size=100).map(str.strip).filter(bool)


def _plan(step_ids: list[SubQuestionId]) -> Plan:
    return Plan(
        id=new_plan_id(),
        run_id=new_run_id(),
        revision=1,
        status=PlanStatus.APPROVED,
        created_by=PlanAuthor.PLANNER,
        created_at=utc_now(),
        steps=tuple(
            SubQuestion(
                id=step_id,
                text="A generated sub-question.",
                rationale="Generated: the shape of the report is what is under test.",
            )
            for step_id in step_ids
        ),
    )


def _source(source_id: SourceId) -> Source:
    return Source(
        id=source_id,
        url="https://example.org/generated",
        title="A generated document",
        retrieved_at=utc_now(),
        content_sha256=DIGEST,
    )


def _fact(fact_id: FactId, step_id: SubQuestionId, source_id: SourceId) -> Fact:
    return Fact(
        id=fact_id,
        sub_question_id=step_id,
        source_id=source_id,
        statement="A generated statement.",
        quote="A generated quote.",
        confidence=Confidence.MEDIUM,
        extracted_at=utc_now(),
    )


def _rebuild(
    report: Report,
    *,
    sources: tuple[Source, ...] | None = None,
    facts: tuple[Fact, ...] | None = None,
) -> Report:
    return Report(
        run_id=report.run_id,
        question=report.question,
        plan=report.plan,
        sources=report.sources if sources is None else sources,
        facts=report.facts if facts is None else facts,
        claims=report.claims,
        generated_at=report.generated_at,
    )


@st.composite
def coherent_reports(draw: st.DrawFn) -> Report:
    """Assemble a report in which every source and every fact is actually referenced.

    Sources and steps are spread across the facts by position rather than drawn per fact,
    so that nothing in the report is unreferenced. That is what makes "remove any one of
    them" a fair test: with an orphan source in the mix, removing it would prove nothing.
    """
    fact_ids = draw(st.lists(FACT_IDS, min_size=1, max_size=6, unique=True))
    source_ids = draw(st.lists(SOURCE_IDS, min_size=1, max_size=len(fact_ids), unique=True))
    step_ids = draw(st.lists(STEP_IDS, min_size=1, max_size=len(fact_ids), unique=True))
    claim_ids = draw(st.lists(CLAIM_IDS, min_size=1, max_size=3, unique=True))

    facts = tuple(
        _fact(fact_id, step_ids[index % len(step_ids)], source_ids[index % len(source_ids)])
        for index, fact_id in enumerate(fact_ids)
    )
    # The first claim cites every fact, so no fact is an orphan; the rest are unverified.
    claims = tuple(
        Claim(id=claim_id, text="A generated claim.", supported_by=tuple(fact_ids))
        if index == 0
        else Claim(
            id=claim_id,
            text="A generated claim.",
            unverified_reason="Generated: nothing was found for this one.",
        )
        for index, claim_id in enumerate(claim_ids)
    )

    return Report(
        run_id=new_run_id(),
        question=QUESTION,
        plan=_plan(step_ids),
        sources=tuple(_source(source_id) for source_id in source_ids),
        facts=facts,
        claims=claims,
        generated_at=utc_now(),
    )


@given(coherent_reports())
def test_any_coherent_report_is_accepted(report: Report) -> None:
    assert len(report.claims) >= 1
    assert {fact.id for fact in report.facts} == set(report.claims[0].supported_by)


@given(coherent_reports(), st.data())
def test_removing_any_cited_fact_breaks_the_report(report: Report, data: st.DataObject) -> None:
    index = data.draw(st.integers(min_value=0, max_value=len(report.facts) - 1))
    remaining = tuple(fact for position, fact in enumerate(report.facts) if position != index)
    with pytest.raises(ReferentialIntegrityError):
        _rebuild(report, facts=remaining)


@given(coherent_reports(), st.data())
def test_removing_any_referenced_source_breaks_the_report(
    report: Report, data: st.DataObject
) -> None:
    index = data.draw(st.integers(min_value=0, max_value=len(report.sources) - 1))
    remaining = tuple(source for position, source in enumerate(report.sources) if position != index)
    with pytest.raises(ReferentialIntegrityError):
        _rebuild(report, sources=remaining)


@given(
    CLAIM_IDS,
    st.lists(FACT_IDS, max_size=3, unique=True),
    st.one_of(st.none(), REASONS),
)
def test_a_claim_is_accepted_exactly_when_it_is_cited_or_explicitly_unverified(
    claim_id: ClaimId,
    supported_by: list[FactId],
    unverified_reason: str | None,
) -> None:
    # Design principle 5 as an exclusive or: evidence, or a stated absence of it, never
    # both and never neither.
    coherent = bool(supported_by) != (unverified_reason is not None)

    if coherent:
        claim = Claim(
            id=claim_id,
            text="A generated claim.",
            supported_by=tuple(supported_by),
            unverified_reason=unverified_reason,
        )
        assert claim.is_unverified is (not supported_by)
    else:
        with pytest.raises(ValidationError):
            Claim(
                id=claim_id,
                text="A generated claim.",
                supported_by=tuple(supported_by),
                unverified_reason=unverified_reason,
            )
