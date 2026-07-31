"""The binding a fact cannot be built without."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.confidence import Confidence
from researchmind.core.fact import MAX_QUOTE_LENGTH, MAX_STATEMENT_LENGTH, Fact
from researchmind.core.ids import (
    new_fact_id,
    new_source_id,
    new_sub_question_id,
)

STATEMENT = "Singapore's stablecoin framework took effect in 2025."
QUOTE = "The framework came into force on 1 January 2025."


def _payload(**overrides: object) -> dict[str, object]:
    # Facts are built through model_validate rather than the constructor so that a test
    # can omit a required field, which keyword arguments would not let it do.
    payload: dict[str, object] = {
        "id": new_fact_id(),
        "sub_question_id": new_sub_question_id(),
        "source_id": new_source_id(),
        "statement": STATEMENT,
        "quote": QUOTE,
        "confidence": Confidence.HIGH,
        "extracted_at": utc_now(),
    }
    payload.update(overrides)
    return payload


def _fact(**overrides: object) -> Fact:
    return Fact.model_validate(_payload(**overrides))


def test_a_fact_carries_its_statement_quote_and_confidence() -> None:
    fact = _fact()
    assert fact.statement == STATEMENT
    assert fact.quote == QUOTE
    assert fact.confidence is Confidence.HIGH


@pytest.mark.parametrize("field", ["source_id", "quote", "confidence", "sub_question_id"])
def test_the_binding_is_not_optional(field: str) -> None:
    # The rule stated in executor - a fact without its binding does not exist - is
    # enforced by the type, not by the caller remembering it.
    payload = _payload()
    del payload[field]
    with pytest.raises(ValidationError):
        Fact.model_validate(payload)


def test_a_fact_cannot_be_left_without_a_source() -> None:
    # "No source found" is the absence of a fact, expressed in the report as a claim with
    # nothing behind it. It is never a fact whose source is null.
    with pytest.raises(ValidationError):
        _fact(source_id=None)


@pytest.mark.parametrize("quote", ["", "   "])
def test_a_fact_without_a_quote_is_rejected(quote: str) -> None:
    with pytest.raises(ValidationError):
        _fact(quote=quote)


@pytest.mark.parametrize("statement", ["", "  "])
def test_a_fact_without_a_statement_is_rejected(statement: str) -> None:
    with pytest.raises(ValidationError):
        _fact(statement=statement)


def test_a_quote_the_size_of_the_document_is_rejected() -> None:
    assert len(_fact(quote="q" * MAX_QUOTE_LENGTH).quote) == MAX_QUOTE_LENGTH
    with pytest.raises(ValidationError):
        _fact(quote="q" * (MAX_QUOTE_LENGTH + 1))


def test_a_statement_past_the_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _fact(statement="s" * (MAX_STATEMENT_LENGTH + 1))


def test_the_extraction_instant_must_be_aware() -> None:
    with pytest.raises(ValidationError):
        _fact(extracted_at=datetime(2026, 7, 28, 12, 0))  # noqa: DTZ001


def test_confidence_is_a_level_not_a_probability() -> None:
    with pytest.raises(ValidationError):
        _fact(confidence=0.83)


def test_a_fact_carries_no_offsets_into_the_document() -> None:
    # Decision recorded in core.fact: binding is by containment, not by position, so that
    # a change in how documents are parsed cannot silently invalidate stored evidence.
    with pytest.raises(ValidationError):
        _fact(quote_start=0, quote_end=len(QUOTE))
