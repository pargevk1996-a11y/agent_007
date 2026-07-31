"""The obligation a verdict of supported carries with it."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.ids import new_source_id
from researchmind.core.review import MAX_SUPPORTING_QUOTE_LENGTH, Review, Verdict

QUOTE = "The framework came into force on 1 January 2025."


def test_a_supported_verdict_carries_its_evidence() -> None:
    source_id = new_source_id()
    review = Review(
        verdict=Verdict.SUPPORTED,
        supporting_quote=QUOTE,
        source_id=source_id,
        reviewed_at=utc_now(),
    )
    assert review.verdict is Verdict.SUPPORTED
    assert review.supporting_quote == QUOTE
    assert review.source_id == source_id


def test_approval_without_a_quote_is_rejected() -> None:
    # ADR-0006: the obligation to quote is the anti-hallucination mechanism, so a critic
    # that approves without producing the span cannot be represented.
    with pytest.raises(ValidationError, match="must quote the span"):
        Review(verdict=Verdict.SUPPORTED, source_id=new_source_id(), reviewed_at=utc_now())


def test_approval_without_a_named_source_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must name the source"):
        Review(verdict=Verdict.SUPPORTED, supporting_quote=QUOTE, reviewed_at=utc_now())


@pytest.mark.parametrize(
    "verdict",
    [Verdict.PARTIALLY_SUPPORTED, Verdict.CONTRADICTED, Verdict.NO_SOURCE],
)
def test_every_other_verdict_may_stand_without_a_quote(verdict: Verdict) -> None:
    # The asymmetry is deliberate: approving without evidence is the failure being
    # designed out, and a critic that disagrees is not obliged to find a span.
    assert Review(verdict=verdict, reviewed_at=utc_now()).supporting_quote is None


def test_a_contradicting_verdict_may_still_quote() -> None:
    review = Review(
        verdict=Verdict.CONTRADICTED,
        supporting_quote=QUOTE,
        source_id=new_source_id(),
        reviewed_at=utc_now(),
    )
    assert review.supporting_quote == QUOTE


@pytest.mark.parametrize("quote", ["", "   "])
def test_a_quote_that_is_blank_is_rejected(quote: str) -> None:
    with pytest.raises(ValidationError):
        Review(
            verdict=Verdict.CONTRADICTED,
            supporting_quote=quote,
            reviewed_at=utc_now(),
        )


def test_a_quote_past_the_ceiling_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Review(
            verdict=Verdict.CONTRADICTED,
            supporting_quote="q" * (MAX_SUPPORTING_QUOTE_LENGTH + 1),
            reviewed_at=utc_now(),
        )


def test_the_review_instant_must_be_aware() -> None:
    with pytest.raises(ValidationError):
        Review(verdict=Verdict.NO_SOURCE, reviewed_at=datetime(2026, 7, 28, 12, 0))  # noqa: DTZ001


def test_verdicts_render_as_their_labels() -> None:
    assert str(Verdict.PARTIALLY_SUPPORTED) == "partially_supported"
    assert str(Verdict.NO_SOURCE) == "no_source"
