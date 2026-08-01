"""The structured contract: what a caller may ask for, and what may come back.

The type parameter is checked by ``mypy --strict`` rather than at runtime; these tests
cover the invariant that types cannot express — when a value is allowed to exist.
"""

from datetime import timedelta

import pytest
from pydantic import Field, ValidationError

from researchmind.core.base import DomainModel
from researchmind.core.ids import new_call_id
from researchmind.core.tokens import TokenUsage
from researchmind.providers.completion import Message, Role, StopReason
from researchmind.providers.structured import StructuredRequest, StructuredResult


class Finding(DomainModel):
    """A small extraction target, with a bound the vendor's schema cannot enforce."""

    headline: str = Field(min_length=1, max_length=80)


def _request() -> StructuredRequest[Finding]:
    return StructuredRequest[Finding](
        call_id=new_call_id(),
        model="claude-opus-5",
        messages=(Message(role=Role.USER, content="Summarise it."),),
        max_tokens=512,
        output_schema=Finding,
    )


def _result(
    *, stop_reason: StopReason, value: Finding | None = None, text: str = "{}"
) -> StructuredResult[Finding]:
    return StructuredResult[Finding](
        call_id=new_call_id(),
        model="claude-opus-5",
        text=text,
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        latency=timedelta(milliseconds=800),
        value=value,
    )


def test_a_structured_request_is_a_completion_request_with_a_schema() -> None:
    request = _request()
    assert request.output_schema is Finding
    assert request.max_tokens == 512
    assert request.messages[0].role is Role.USER


def test_the_exchange_rules_are_inherited_rather_than_restated() -> None:
    # Whatever `complete` refuses to send, `complete_structured` refuses too.
    with pytest.raises(ValidationError):
        StructuredRequest[Finding](
            call_id=new_call_id(),
            model="claude-opus-5",
            messages=(Message(role=Role.ASSISTANT, content="I will answer first."),),
            max_tokens=512,
            output_schema=Finding,
        )


def test_a_schema_that_is_not_a_domain_model_is_refused() -> None:
    with pytest.raises(ValidationError):
        StructuredRequest[Finding](
            call_id=new_call_id(),
            model="claude-opus-5",
            messages=(Message(role=Role.USER, content="Summarise it."),),
            max_tokens=512,
            output_schema=int,  # type: ignore[arg-type]  # the point of the test
        )


def test_a_completed_generation_may_carry_a_value() -> None:
    result = _result(stop_reason=StopReason.END_TURN, value=Finding(headline="MiCA applies."))
    assert result.extracted
    assert result.value is not None
    assert result.value.headline == "MiCA applies."


def test_a_completed_generation_may_also_carry_nothing() -> None:
    # The schema sent to a vendor does not carry every constraint our types impose, so a
    # model can end its turn with an object that will not validate. Forbidding this would
    # make the outcome the retry policy exists for unrepresentable.
    result = _result(stop_reason=StopReason.END_TURN, text='{"headline": ""}')
    assert not result.extracted
    assert result.value is None
    assert result.usage.output_tokens == 20


@pytest.mark.parametrize(
    "stop_reason",
    [
        StopReason.MAX_TOKENS,
        StopReason.REFUSAL,
        StopReason.CONTEXT_WINDOW_EXCEEDED,
        StopReason.STOP_SEQUENCE,
    ],
)
def test_a_generation_that_did_not_finish_cannot_carry_a_value(stop_reason: StopReason) -> None:
    # A truncated document parsed into an object would be an object built from half a
    # sentence, and nothing downstream could tell.
    with pytest.raises(ValidationError):
        _result(stop_reason=stop_reason, value=Finding(headline="Half a thought"))


@pytest.mark.parametrize(
    "stop_reason",
    [StopReason.MAX_TOKENS, StopReason.REFUSAL, StopReason.CONTEXT_WINDOW_EXCEEDED],
)
def test_a_generation_that_did_not_finish_still_reports_what_it_cost(
    stop_reason: StopReason,
) -> None:
    result = _result(stop_reason=stop_reason)
    assert not result.extracted
    assert result.usage.total == 120


def test_a_result_is_a_completion_result() -> None:
    # Everything that accounts for a call accounts for this one the same way.
    result = _result(stop_reason=StopReason.END_TURN)
    assert result.latency == timedelta(milliseconds=800)
    assert result.model == "claude-opus-5"
    assert result.text == "{}"
