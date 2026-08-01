"""Prices, the cost they produce, and the reservation taken before a call is made.

The models here are fictional on purpose. These tests exercise the arithmetic, and naming a
real model would state a second, competing claim about what that model costs — the shipped
one lives in ``researchmind.providers.prices`` and is asserted in ``test_prices.py``. A
vendor changing a rate should break the price data and nothing else.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.cost import CallKind
from researchmind.core.ids import new_call_id, new_run_id, new_sub_question_id
from researchmind.core.money import Money
from researchmind.core.tokens import TokenUsage
from researchmind.providers.completion import CompletionResult, StopReason
from researchmind.providers.errors import ProviderError, UnknownModelPriceError
from researchmind.providers.pricing import ModelPrice, PriceList

DEAR = ModelPrice.from_dollars_per_mtok(
    provider="anthropic",
    model="model-dear",
    input_dollars_per_mtok=Decimal("15.00"),
    cached_input_dollars_per_mtok=Decimal("1.50"),
    output_dollars_per_mtok=Decimal("75.00"),
)
CHEAP = ModelPrice.from_dollars_per_mtok(
    provider="anthropic",
    model="model-cheap",
    input_dollars_per_mtok=Decimal("1.00"),
    cached_input_dollars_per_mtok=Decimal("0.10"),
    output_dollars_per_mtok=Decimal("5.00"),
)


def _price_list(*prices: ModelPrice, version: str = "2026-08-01") -> PriceList:
    return PriceList(
        version=version,
        effective_from=datetime(2026, 8, 1, tzinfo=UTC),
        prices=prices if prices else (DEAR, CHEAP),
    )


def _result(
    *,
    model: str = "model-dear",
    usage: TokenUsage | None = None,
    latency: timedelta = timedelta(milliseconds=1200),
) -> CompletionResult:
    return CompletionResult(
        call_id=new_call_id(),
        model=model,
        text="Here is the comparison.",
        stop_reason=StopReason.END_TURN,
        usage=usage if usage is not None else TokenUsage(input_tokens=1000, output_tokens=500),
        latency=latency,
    )


def test_a_quoted_rate_becomes_an_exact_price_per_token() -> None:
    # A dollar per million tokens is a thousand nanodollars per token, with nothing to
    # round: $15.00 per million is 15,000, and $1.50 is 1,500.
    assert DEAR.per_input_token == Money(nanodollars=15_000)
    assert DEAR.per_cached_input_token == Money(nanodollars=1_500)
    assert DEAR.per_output_token == Money(nanodollars=75_000)


def test_the_smallest_quoted_rates_survive_the_conversion() -> None:
    # A tenth of a dollar per million tokens is a hundred nanodollars per token: three
    # orders of magnitude above the resolution of the type.
    assert CHEAP.per_cached_input_token == Money(nanodollars=100)


def test_a_rate_may_be_quoted_as_a_string_or_a_whole_number() -> None:
    price = ModelPrice.from_dollars_per_mtok(
        provider="vllm",
        model="local",
        input_dollars_per_mtok="0.25",
        cached_input_dollars_per_mtok=0,
        output_dollars_per_mtok=1,
    )
    assert price.per_input_token == Money(nanodollars=250)
    assert price.per_cached_input_token == Money(nanodollars=0)
    assert price.per_output_token == Money(nanodollars=1000)


def test_a_free_model_is_priced_at_zero_rather_than_refused() -> None:
    # A self-hosted endpoint bills nothing per token, and that is a price, not an absence.
    price = ModelPrice(
        provider="vllm",
        model="qwen-3",
        per_input_token=Money(nanodollars=0),
        per_cached_input_token=Money(nanodollars=0),
        per_output_token=Money(nanodollars=0),
    )
    assert price.charge_for(TokenUsage(input_tokens=10**6, output_tokens=10**6)) == Money(
        nanodollars=0
    )


def test_a_negative_price_is_refused() -> None:
    with pytest.raises(ValidationError):
        ModelPrice(
            provider="anthropic",
            model="model-dear",
            per_input_token=Money(nanodollars=-1),
            per_cached_input_token=Money(nanodollars=0),
            per_output_token=Money(nanodollars=0),
        )


def test_usage_is_charged_by_the_kind_of_token_it_reports() -> None:
    usage = TokenUsage(input_tokens=1000, cached_input_tokens=2000, output_tokens=100)
    expected = 1000 * 15_000 + 2000 * 1_500 + 100 * 75_000
    assert DEAR.charge_for(usage) == Money(nanodollars=expected)


def test_cached_tokens_are_not_charged_at_the_full_input_rate() -> None:
    # The saving prompt caching exists for has to be visible in the number, or the split
    # in TokenUsage measures nothing.
    cached = DEAR.charge_for(
        TokenUsage(input_tokens=0, cached_input_tokens=10_000, output_tokens=0)
    )
    uncached = DEAR.charge_for(TokenUsage(input_tokens=10_000, output_tokens=0))
    assert cached < uncached


def test_the_reservation_prices_the_whole_of_max_tokens() -> None:
    # ADR-0004: the bound assumes generation runs to the limit, because it is taken before
    # anything is known about how much of it will be used.
    assert DEAR.upper_bound(input_tokens=1000, max_tokens=4096) == Money(
        nanodollars=1000 * 15_000 + 4096 * 75_000
    )


def test_the_reservation_covers_what_the_call_turns_out_to_cost() -> None:
    reserved = DEAR.upper_bound(input_tokens=1000, max_tokens=4096)
    actual = DEAR.charge_for(TokenUsage(input_tokens=1000, output_tokens=4096))
    assert reserved == actual

    cheaper = DEAR.charge_for(TokenUsage(input_tokens=1000, output_tokens=12))
    assert cheaper < reserved


def test_the_reservation_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        DEAR.upper_bound(input_tokens=-1, max_tokens=100)


def test_a_price_list_finds_a_model_it_prices() -> None:
    assert _price_list().price_for("anthropic", "model-cheap") is CHEAP


def test_an_unpriced_model_raises_rather_than_costing_nothing() -> None:
    with pytest.raises(UnknownModelPriceError) as caught:
        _price_list().price_for("anthropic", "model-unlisted")

    assert caught.value.provider == "anthropic"
    assert caught.value.model == "model-unlisted"
    assert caught.value.price_version == "2026-08-01"


def test_an_unpriced_model_is_catchable_where_the_call_itself_is() -> None:
    with pytest.raises(ProviderError):
        _price_list().price_for("openai", "model-dear")


def test_the_same_model_from_two_providers_is_two_prices() -> None:
    # Provider is part of the key: the same open-weights model is served at different
    # rates by different hosts.
    hosted = ModelPrice.from_dollars_per_mtok(
        provider="vllm",
        model="model-dear",
        input_dollars_per_mtok=1,
        cached_input_dollars_per_mtok=0,
        output_dollars_per_mtok=2,
    )
    prices = _price_list(DEAR, hosted)
    assert prices.price_for("anthropic", "model-dear") is DEAR
    assert prices.price_for("vllm", "model-dear") is hosted


def test_a_model_priced_twice_is_refused() -> None:
    with pytest.raises(ValidationError):
        _price_list(DEAR, DEAR)


def test_a_price_list_that_prices_nothing_is_refused() -> None:
    # An empty list prices every model at "unknown", which is a configuration mistake
    # worth catching where it is written rather than at the first call.
    with pytest.raises(ValidationError):
        PriceList(version="empty", effective_from=utc_now(), prices=())


def test_a_completion_becomes_a_cost_stamped_with_the_version_that_priced_it() -> None:
    result = _result(usage=TokenUsage(input_tokens=1000, cached_input_tokens=200, output_tokens=50))
    run_id = new_run_id()
    incurred_at = utc_now()

    cost = _price_list().cost_for_completion(
        result,
        provider="anthropic",
        run_id=run_id,
        incurred_at=incurred_at,
    )

    assert cost.call_id == result.call_id
    assert cost.run_id == run_id
    assert cost.sub_question_id is None
    assert cost.kind is CallKind.LLM
    assert cost.provider == "anthropic"
    assert cost.model == "model-dear"
    assert cost.tokens == result.usage
    assert cost.amount == DEAR.charge_for(result.usage)
    assert cost.latency == result.latency
    assert cost.price_version == "2026-08-01"
    assert cost.incurred_at == incurred_at


def test_a_cost_can_be_attributed_to_the_sub_question_that_caused_it() -> None:
    sub_question_id = new_sub_question_id()
    cost = _price_list().cost_for_completion(
        _result(),
        provider="anthropic",
        run_id=new_run_id(),
        incurred_at=utc_now(),
        sub_question_id=sub_question_id,
    )
    assert cost.sub_question_id == sub_question_id


def test_the_model_that_served_the_call_is_the_one_that_is_priced() -> None:
    # An alias resolved, or a deployment was pointed elsewhere. Pricing the requested name
    # would attribute the spend to a model that did no work.
    cost = _price_list().cost_for_completion(
        _result(model="model-cheap"),
        provider="anthropic",
        run_id=new_run_id(),
        incurred_at=utc_now(),
    )
    assert cost.model == "model-cheap"
    assert cost.amount == CHEAP.charge_for(TokenUsage(input_tokens=1000, output_tokens=500))


def test_a_completion_from_an_unpriced_model_produces_no_cost_row() -> None:
    with pytest.raises(UnknownModelPriceError):
        _price_list().cost_for_completion(
            _result(model="model-unlisted"),
            provider="anthropic",
            run_id=new_run_id(),
            incurred_at=utc_now(),
        )
