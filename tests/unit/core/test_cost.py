"""What a cost record must carry, and the calls it must match."""

from collections.abc import Callable
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.cost import CallKind, Cost
from researchmind.core.ids import (
    SubQuestionId,
    new_call_id,
    new_run_id,
    new_sub_question_id,
)
from researchmind.core.money import Money
from researchmind.core.tokens import TokenUsage

USAGE = TokenUsage(input_tokens=1200, cached_input_tokens=8000, output_tokens=300)


def _cost(  # noqa: PLR0913  # a builder mirrors the model, and Cost has that many fields
    *,
    kind: CallKind = CallKind.LLM,
    tokens: TokenUsage | None = USAGE,
    amount: Money | None = None,
    latency: timedelta = timedelta(milliseconds=850),
    provider: str = "anthropic",
    model: str = "claude-opus-5",
    price_version: str = "2026-07-01",
    sub_question_id: SubQuestionId | None = None,
    incurred_at: datetime | None = None,
) -> Cost:
    return Cost(
        call_id=new_call_id(),
        run_id=new_run_id(),
        sub_question_id=sub_question_id,
        kind=kind,
        provider=provider,
        model=model,
        tokens=tokens,
        amount=amount if amount is not None else Money.from_dollars("0.0123"),
        latency=latency,
        price_version=price_version,
        incurred_at=incurred_at if incurred_at is not None else utc_now(),
    )


def test_an_llm_call_records_its_tokens_and_its_price() -> None:
    cost = _cost()
    assert cost.tokens is not None
    assert cost.tokens.total == 9500
    assert cost.amount == Money.from_dollars("0.0123")
    assert cost.price_version == "2026-07-01"


def test_a_tool_call_carries_no_token_count() -> None:
    cost = _cost(kind=CallKind.TOOL, tokens=None)
    assert cost.tokens is None


def test_a_tool_call_that_reports_tokens_is_rejected() -> None:
    # Zero or otherwise, a token count on a request-priced call is a measurement of
    # something that was never measured.
    with pytest.raises(ValidationError, match="invents a measurement"):
        _cost(kind=CallKind.TOOL, tokens=USAGE)


@pytest.mark.parametrize("kind", [CallKind.LLM, CallKind.EMBEDDING])
def test_a_call_that_spends_tokens_must_report_them(kind: CallKind) -> None:
    with pytest.raises(ValidationError, match="must report its token usage"):
        _cost(kind=kind, tokens=None)


def test_a_negative_cost_is_rejected() -> None:
    # Money permits negatives so a released reservation can be expressed. An incurred cost
    # is a quantity, not a movement, and a negative one would mean a call that paid us.
    with pytest.raises(ValidationError, match="cannot be negative"):
        _cost(amount=Money.from_dollars("-0.01"))


def test_a_cost_of_nothing_is_allowed() -> None:
    # A cached-only call can legitimately round to zero, and a free tool costs nothing.
    assert _cost(amount=Money.from_dollars("0")).amount.nanodollars == 0


def test_a_planner_call_has_no_sub_question() -> None:
    # Planning happens before any sub-question exists, so the middle link of the
    # run-step-call chain is genuinely absent rather than unknown.
    assert _cost().sub_question_id is None


def test_a_call_made_for_a_step_names_it() -> None:
    step_id = new_sub_question_id()
    assert _cost(sub_question_id=step_id).sub_question_id == step_id


def test_a_negative_latency_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _cost(latency=timedelta(milliseconds=-1))


@pytest.mark.parametrize(
    "build",
    [
        lambda: _cost(provider="   "),
        lambda: _cost(model="   "),
        lambda: _cost(price_version="   "),
    ],
    ids=["provider", "model", "price_version"],
)
def test_the_attribution_fields_cannot_be_blank(build: Callable[[], Cost]) -> None:
    # Without these a cost row cannot be traced back to what produced it, which is the
    # whole of design principle 7.
    with pytest.raises(ValidationError):
        build()


def test_the_instant_must_be_aware() -> None:
    with pytest.raises(ValidationError):
        _cost(incurred_at=datetime(2026, 7, 28, 12, 0))  # noqa: DTZ001


def test_call_kinds_render_as_their_labels() -> None:
    assert str(CallKind.EMBEDDING) == "embedding"
    assert str(CallKind.TOOL) == "tool"
