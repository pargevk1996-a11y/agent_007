"""The invariants that only hold across a whole report, and how they fail."""

from hashlib import sha256

import pytest
from pydantic import ValidationError

from researchmind.core.claim import Claim
from researchmind.core.clock import utc_now
from researchmind.core.confidence import Confidence
from researchmind.core.errors import CoreError, ResearchmindError
from researchmind.core.fact import Fact
from researchmind.core.ids import (
    FactId,
    SourceId,
    SubQuestionId,
    new_claim_id,
    new_fact_id,
    new_plan_id,
    new_run_id,
    new_source_id,
    new_sub_question_id,
)
from researchmind.core.plan import Plan, PlanAuthor, PlanStatus, SubQuestion
from researchmind.core.question import ResearchQuestion
from researchmind.core.report import ReferentialIntegrityError, Report
from researchmind.core.review import Review, Verdict
from researchmind.core.source import Source

QUESTION = ResearchQuestion(text="Which jurisdictions issued stablecoin rules in 2025?")
DIGEST = sha256(b"the document as it was parsed").hexdigest()


def _plan_with(step_id: SubQuestionId) -> Plan:
    return Plan(
        id=new_plan_id(),
        run_id=new_run_id(),
        revision=1,
        status=PlanStatus.APPROVED,
        created_by=PlanAuthor.PLANNER,
        created_at=utc_now(),
        steps=(
            SubQuestion(
                id=step_id,
                text="Which regulators published rules in 2025?",
                rationale="Establishes the set the rest of the plan compares.",
            ),
        ),
    )


def _source_with(source_id: SourceId) -> Source:
    return Source(
        id=source_id,
        url="https://example.org/stablecoins-2025",
        title="Stablecoin regulation in 2025",
        retrieved_at=utc_now(),
        content_sha256=DIGEST,
    )


def _fact_with(fact_id: FactId, step_id: SubQuestionId, source_id: SourceId) -> Fact:
    return Fact(
        id=fact_id,
        sub_question_id=step_id,
        source_id=source_id,
        statement="Singapore's framework took effect in 2025.",
        quote="The framework came into force on 1 January 2025.",
        confidence=Confidence.HIGH,
        extracted_at=utc_now(),
    )


def _report(
    *,
    plan: Plan,
    sources: tuple[Source, ...] = (),
    facts: tuple[Fact, ...] = (),
    claims: tuple[Claim, ...],
) -> Report:
    return Report(
        run_id=new_run_id(),
        question=QUESTION,
        plan=plan,
        sources=sources,
        facts=facts,
        claims=claims,
        generated_at=utc_now(),
    )


def _coherent() -> Report:
    step_id, source_id, fact_id = new_sub_question_id(), new_source_id(), new_fact_id()
    return _report(
        plan=_plan_with(step_id),
        sources=(_source_with(source_id),),
        facts=(_fact_with(fact_id, step_id, source_id),),
        claims=(Claim(id=new_claim_id(), text="A cited claim.", supported_by=(fact_id,)),),
    )


def test_a_coherent_report_is_accepted() -> None:
    report = _coherent()
    assert len(report.claims) == 1
    assert report.claims[0].supported_by[0] == report.facts[0].id


def test_a_report_that_found_nothing_is_still_a_report() -> None:
    # An honest search that turned up nothing produces claims that are every one of them
    # explicitly unverified. Sources and facts are legitimately empty.
    report = _report(
        plan=_plan_with(new_sub_question_id()),
        claims=(
            Claim(
                id=new_claim_id(),
                text="No commencement date could be established.",
                unverified_reason="No source states a date.",
            ),
        ),
    )
    assert report.sources == ()
    assert report.facts == ()
    assert report.claims[0].is_unverified


def test_a_report_that_states_nothing_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _report(plan=_plan_with(new_sub_question_id()), claims=())


def test_a_fact_citing_an_absent_source_is_refused() -> None:
    step_id = new_sub_question_id()
    fact = _fact_with(new_fact_id(), step_id, new_source_id())
    with pytest.raises(ReferentialIntegrityError, match="which the report omits"):
        _report(
            plan=_plan_with(step_id),
            facts=(fact,),
            claims=(Claim(id=new_claim_id(), text="A claim.", supported_by=(fact.id,)),),
        )


def test_a_fact_attributed_to_a_step_outside_the_plan_is_refused() -> None:
    # ADR-0002 exists to make this attribution reliable; a fact pointing at a step that
    # was never executed makes the trace untrue.
    source_id = new_source_id()
    fact = _fact_with(new_fact_id(), new_sub_question_id(), source_id)
    with pytest.raises(ReferentialIntegrityError, match="not a step of the executed plan"):
        _report(
            plan=_plan_with(new_sub_question_id()),
            sources=(_source_with(source_id),),
            facts=(fact,),
            claims=(Claim(id=new_claim_id(), text="A claim.", supported_by=(fact.id,)),),
        )


def test_a_claim_citing_an_absent_fact_is_refused() -> None:
    with pytest.raises(ReferentialIntegrityError, match="cites fact"):
        _report(
            plan=_plan_with(new_sub_question_id()),
            claims=(Claim(id=new_claim_id(), text="A claim.", supported_by=(new_fact_id(),)),),
        )


def test_a_review_quoting_an_absent_source_is_refused() -> None:
    with pytest.raises(ReferentialIntegrityError, match="quotes source"):
        _report(
            plan=_plan_with(new_sub_question_id()),
            claims=(
                Claim(
                    id=new_claim_id(),
                    text="A claim.",
                    unverified_reason="Nothing was found.",
                    review=Review(
                        verdict=Verdict.SUPPORTED,
                        supporting_quote="A span from somewhere.",
                        source_id=new_source_id(),
                        reviewed_at=utc_now(),
                    ),
                ),
            ),
        )


def test_a_repeated_source_identifier_is_refused() -> None:
    source_id = new_source_id()
    with pytest.raises(ReferentialIntegrityError, match="same source identifier"):
        _report(
            plan=_plan_with(new_sub_question_id()),
            sources=(_source_with(source_id), _source_with(source_id)),
            claims=(Claim(id=new_claim_id(), text="A claim.", unverified_reason="Nothing found."),),
        )


def test_a_repeated_fact_identifier_is_refused() -> None:
    step_id, source_id, fact_id = new_sub_question_id(), new_source_id(), new_fact_id()
    with pytest.raises(ReferentialIntegrityError, match="same fact identifier"):
        _report(
            plan=_plan_with(step_id),
            sources=(_source_with(source_id),),
            facts=(
                _fact_with(fact_id, step_id, source_id),
                _fact_with(fact_id, step_id, source_id),
            ),
            claims=(Claim(id=new_claim_id(), text="A claim.", unverified_reason="Nothing found."),),
        )


def test_a_repeated_claim_identifier_is_refused() -> None:
    claim_id = new_claim_id()
    with pytest.raises(ReferentialIntegrityError, match="same claim identifier"):
        _report(
            plan=_plan_with(new_sub_question_id()),
            claims=(
                Claim(id=claim_id, text="One claim.", unverified_reason="Nothing found."),
                Claim(id=claim_id, text="Another claim.", unverified_reason="Nothing found."),
            ),
        )


def test_a_referential_failure_is_not_a_validation_error() -> None:
    # The consequence of raising a CoreError from inside a validator, made explicit: this
    # exception travels past `except ValidationError`. A caller wanting both must catch
    # ResearchmindError as well.
    with pytest.raises(ReferentialIntegrityError) as caught:
        _report(
            plan=_plan_with(new_sub_question_id()),
            claims=(Claim(id=new_claim_id(), text="A claim.", supported_by=(new_fact_id(),)),),
        )
    assert not isinstance(caught.value, ValidationError)
    assert isinstance(caught.value, CoreError)
    assert isinstance(caught.value, ResearchmindError)


def test_the_report_is_identified_by_its_run() -> None:
    # There is no ReportId: a run produces one report, and inventing a second identity
    # would only raise the question of which belongs to which.
    report = _coherent()
    assert not hasattr(report, "id")
    assert report.run_id is not None
