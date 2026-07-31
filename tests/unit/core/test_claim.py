"""Design principle 5, as an invariant: cited, or unverified in so many words."""

import pytest
from pydantic import ValidationError

from researchmind.core.claim import MAX_CLAIM_LENGTH, MAX_UNVERIFIED_REASON_LENGTH, Claim
from researchmind.core.clock import utc_now
from researchmind.core.ids import new_claim_id, new_fact_id, new_source_id
from researchmind.core.review import Review, Verdict

TEXT = "Singapore's stablecoin framework took effect in 2025."
REASON = "No source was found that states a commencement date."


def test_a_cited_claim_names_the_facts_behind_it() -> None:
    fact_id = new_fact_id()
    claim = Claim(id=new_claim_id(), text=TEXT, supported_by=(fact_id,))
    assert claim.supported_by == (fact_id,)
    assert claim.unverified_reason is None
    assert not claim.is_unverified


def test_an_unverified_claim_says_why() -> None:
    claim = Claim(id=new_claim_id(), text=TEXT, unverified_reason=REASON)
    assert claim.supported_by == ()
    assert claim.unverified_reason == REASON
    assert claim.is_unverified


def test_a_claim_with_neither_citation_nor_reason_is_rejected() -> None:
    # The silent gap the principle exists to forbid: an empty list asserts nothing, so it
    # cannot stand in for the statement that nothing was found.
    with pytest.raises(ValidationError, match="must say why it is unverified"):
        Claim(id=new_claim_id(), text=TEXT)


def test_a_claim_cannot_be_both_cited_and_unverified() -> None:
    with pytest.raises(ValidationError, match="cannot also be marked unverified"):
        Claim(
            id=new_claim_id(),
            text=TEXT,
            supported_by=(new_fact_id(),),
            unverified_reason=REASON,
        )


def test_a_claim_cannot_cite_the_same_fact_twice() -> None:
    fact_id = new_fact_id()
    with pytest.raises(ValidationError, match="same fact more than once"):
        Claim(id=new_claim_id(), text=TEXT, supported_by=(fact_id, fact_id))


@pytest.mark.parametrize("text", ["", "   "])
def test_a_claim_without_text_is_rejected(text: str) -> None:
    with pytest.raises(ValidationError):
        Claim(id=new_claim_id(), text=text, unverified_reason=REASON)


def test_a_claim_past_the_length_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Claim(id=new_claim_id(), text="c" * (MAX_CLAIM_LENGTH + 1), unverified_reason=REASON)


@pytest.mark.parametrize("reason", ["", "   "])
def test_a_blank_reason_does_not_count_as_a_reason(reason: str) -> None:
    with pytest.raises(ValidationError):
        Claim(id=new_claim_id(), text=TEXT, unverified_reason=reason)


def test_a_reason_past_the_length_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Claim(
            id=new_claim_id(),
            text=TEXT,
            unverified_reason="r" * (MAX_UNVERIFIED_REASON_LENGTH + 1),
        )


def test_an_unreviewed_claim_is_not_the_same_as_an_unfavourable_verdict() -> None:
    # None means the critic has not run. It must not be confused with NO_SOURCE, which is
    # the critic having run and found the cited evidence beside the point.
    unreviewed = Claim(id=new_claim_id(), text=TEXT, unverified_reason=REASON)
    reviewed = Claim(
        id=new_claim_id(),
        text=TEXT,
        unverified_reason=REASON,
        review=Review(verdict=Verdict.NO_SOURCE, reviewed_at=utc_now()),
    )
    assert unreviewed.review is None
    assert reviewed.review is not None
    assert reviewed.review.verdict is Verdict.NO_SOURCE


def test_an_unverified_claim_may_still_carry_a_review() -> None:
    # The two statements are different: the synthesiser found nothing to cite, and the
    # critic agrees there is no source. Both can be recorded.
    claim = Claim(
        id=new_claim_id(),
        text=TEXT,
        unverified_reason=REASON,
        review=Review(
            verdict=Verdict.CONTRADICTED,
            supporting_quote="The framework was postponed.",
            source_id=new_source_id(),
            reviewed_at=utc_now(),
        ),
    )
    assert claim.is_unverified
    assert claim.review is not None
