"""The Anthropic price data, and choosing the list that was in effect.

These tests assert the published rates. When a vendor changes a price, the intended failure
is here and nowhere else: the mechanism tests in ``test_pricing.py`` deliberately use
fictional models so that a rate change cannot break arithmetic that has nothing to do with
it.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.cost import CallKind
from researchmind.core.ids import new_call_id, new_run_id
from researchmind.core.money import Money
from researchmind.core.tokens import TokenUsage
from researchmind.providers.anthropic_adapter import PROVIDER_NAME
from researchmind.providers.completion import CompletionResult, StopReason
from researchmind.providers.errors import (
    ProviderError,
    UnknownModelPriceError,
    UnpricedInstantError,
)
from researchmind.providers.prices import (
    ANTHROPIC_2026_08,
    ANTHROPIC_2026_09,
    ANTHROPIC_PRICES,
)
from researchmind.providers.pricing import ModelPrice, PriceList, PriceListHistory

AUGUST = datetime(2026, 8, 15, tzinfo=UTC)
SEPTEMBER = datetime(2026, 9, 15, tzinfo=UTC)
LAST_INTRODUCTORY_INSTANT = datetime(2026, 8, 31, 23, 59, 59, tzinfo=UTC)
FIRST_STANDARD_INSTANT = datetime(2026, 9, 1, tzinfo=UTC)


def _price(model: str, dollars_in: str) -> ModelPrice:
    return ModelPrice.from_dollars_per_mtok(
        provider=PROVIDER_NAME,
        model=model,
        input_dollars_per_mtok=dollars_in,
        cached_input_dollars_per_mtok=0,
        output_dollars_per_mtok=0,
    )


def _list(version: str, effective_from: datetime, price: ModelPrice) -> PriceList:
    return PriceList(version=version, effective_from=effective_from, prices=(price,))


def test_the_published_opus_rates_are_what_we_charge_against() -> None:
    # $5.00 and $25.00 per million tokens, exact to the nanodollar.
    opus = ANTHROPIC_2026_08.price_for(PROVIDER_NAME, "claude-opus-5")
    assert opus.per_input_token == Money(nanodollars=5_000)
    assert opus.per_output_token == Money(nanodollars=25_000)


def test_a_cache_read_is_a_tenth_of_the_input_rate() -> None:
    # Derived rather than quoted; if the vendor ever publishes a figure, this is the
    # assertion that has to be revisited rather than silently kept.
    for prices in ANTHROPIC_PRICES.lists:
        for price in prices.prices:
            assert price.per_cached_input_token * 10 == price.per_input_token


def test_the_cheapest_model_is_priced_well_above_the_resolution_of_the_type() -> None:
    # A tenth of a dollar per million tokens is a hundred nanodollars per token.
    haiku = ANTHROPIC_2026_08.price_for(PROVIDER_NAME, "claude-haiku-4-5")
    assert haiku.per_cached_input_token == Money(nanodollars=100)


def test_sonnet_runs_at_its_introductory_rate_in_august() -> None:
    sonnet = ANTHROPIC_2026_08.price_for(PROVIDER_NAME, "claude-sonnet-5")
    assert sonnet.per_input_token == Money(nanodollars=2_000)
    assert sonnet.per_output_token == Money(nanodollars=10_000)


def test_sonnet_returns_to_its_standard_rate_in_september() -> None:
    sonnet = ANTHROPIC_2026_09.price_for(PROVIDER_NAME, "claude-sonnet-5")
    assert sonnet.per_input_token == Money(nanodollars=3_000)
    assert sonnet.per_output_token == Money(nanodollars=15_000)


def test_only_sonnet_changes_between_the_two_lists() -> None:
    # The second list exists because of one expiring rate. Anything else differing between
    # them is a transcription slip rather than a decision.
    august = {price.model: price for price in ANTHROPIC_2026_08.prices}
    september = {price.model: price for price in ANTHROPIC_2026_09.prices}
    assert august.keys() == september.keys()
    assert {model for model in august if august[model] != september[model]} == {"claude-sonnet-5"}


def test_the_history_answers_what_was_in_effect() -> None:
    assert ANTHROPIC_PRICES.at(AUGUST) is ANTHROPIC_2026_08
    assert ANTHROPIC_PRICES.at(SEPTEMBER) is ANTHROPIC_2026_09


def test_the_introductory_rate_holds_to_its_last_second() -> None:
    assert ANTHROPIC_PRICES.at(LAST_INTRODUCTORY_INSTANT) is ANTHROPIC_2026_08


def test_the_standard_rate_applies_from_its_first_instant() -> None:
    # `effective_from` is inclusive: the list is in effect at the instant it takes effect.
    assert ANTHROPIC_PRICES.at(FIRST_STANDARD_INSTANT) is ANTHROPIC_2026_09


def test_a_later_instant_still_resolves_to_the_latest_list() -> None:
    # A history with no successor keeps applying. That is right until someone forgets to
    # add the next list, which is a fact about our diligence and not about the type.
    assert ANTHROPIC_PRICES.at(datetime(2027, 1, 1, tzinfo=UTC)) is ANTHROPIC_2026_09


def test_an_instant_before_every_list_is_refused() -> None:
    before = datetime(2026, 7, 31, tzinfo=UTC)

    with pytest.raises(UnpricedInstantError) as caught:
        ANTHROPIC_PRICES.at(before)

    assert caught.value.provider == PROVIDER_NAME
    assert caught.value.instant == before
    assert caught.value.retryable is False


def test_an_unpriced_instant_is_catchable_where_the_call_itself_is() -> None:
    with pytest.raises(ProviderError):
        ANTHROPIC_PRICES.at(datetime(2020, 1, 1, tzinfo=UTC))


def test_a_model_we_cannot_use_is_absent_rather_than_priced() -> None:
    # Fable needs a data-retention configuration we have not agreed to. Pricing it would
    # invite a call we would then have to refuse for a different reason.
    with pytest.raises(UnknownModelPriceError):
        ANTHROPIC_2026_08.price_for(PROVIDER_NAME, "claude-fable-5")


def test_a_history_must_run_forward_in_time() -> None:
    with pytest.raises(ValidationError):
        PriceListHistory(
            provider=PROVIDER_NAME,
            lists=(ANTHROPIC_2026_09, ANTHROPIC_2026_08),
        )


def test_two_lists_cannot_take_effect_at_the_same_instant() -> None:
    # Which one applied would depend on tuple order, and no reader of a cost row could see
    # that difference.
    same_day = _list("anthropic-again", ANTHROPIC_2026_08.effective_from, _price("m", "1"))

    with pytest.raises(ValidationError):
        PriceListHistory(provider=PROVIDER_NAME, lists=(ANTHROPIC_2026_08, same_day))


def test_two_lists_cannot_share_a_version() -> None:
    duplicate = _list(
        ANTHROPIC_2026_08.version, datetime(2026, 10, 1, tzinfo=UTC), _price("m", "1")
    )

    with pytest.raises(ValidationError):
        PriceListHistory(provider=PROVIDER_NAME, lists=(ANTHROPIC_2026_08, duplicate))


def test_a_history_refuses_prices_belonging_to_another_provider() -> None:
    foreign = ModelPrice.from_dollars_per_mtok(
        provider="openai",
        model="some-model",
        input_dollars_per_mtok=1,
        cached_input_dollars_per_mtok=0,
        output_dollars_per_mtok=2,
    )
    mixed = PriceList(
        version="mixed", effective_from=datetime(2026, 10, 1, tzinfo=UTC), prices=(foreign,)
    )

    with pytest.raises(ValidationError):
        PriceListHistory(provider=PROVIDER_NAME, lists=(ANTHROPIC_2026_08, mixed))


def test_a_history_must_hold_at_least_one_list() -> None:
    with pytest.raises(ValidationError):
        PriceListHistory(provider=PROVIDER_NAME, lists=())


def test_the_history_prices_what_the_adapter_would_have_returned() -> None:
    # The whole chain: a completion's usage, priced by the list in effect when it happened,
    # recorded against the version that priced it.
    result = CompletionResult(
        call_id=new_call_id(),
        model="claude-opus-5",
        text="Here is the comparison.",
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=10_000, cached_input_tokens=90_000, output_tokens=2_000),
        latency=timedelta(milliseconds=1500),
    )
    incurred_at = utc_now()

    cost = ANTHROPIC_PRICES.at(AUGUST).cost_for_completion(
        result,
        provider=PROVIDER_NAME,
        run_id=new_run_id(),
        incurred_at=incurred_at,
    )

    # 10,000 x 5,000 + 90,000 x 500 + 2,000 x 25,000 nanodollars, which is 14.5 cents.
    assert cost.amount == Money(nanodollars=50_000_000 + 45_000_000 + 50_000_000)
    assert cost.amount.as_dollars() == Money.from_dollars("0.145").as_dollars()
    assert cost.kind is CallKind.LLM
    assert cost.price_version == "anthropic-2026-08"
    assert cost.incurred_at == incurred_at
