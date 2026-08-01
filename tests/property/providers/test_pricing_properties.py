"""Properties of pricing, over generated rates and token counts.

Charging is arithmetic, and the claims worth making about it are algebraic: that it agrees
with token accumulation, that more tokens never cost less, and that the reservation ADR-0004
takes before a call is genuinely an upper bound on what the call turns out to cost. Three
worked examples cannot say any of that.
"""

from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from researchmind.core.money import Money
from researchmind.core.tokens import TokenUsage
from researchmind.providers.pricing import ModelPrice

# Bounded so that the products stay inside the bigint range Money is held to: a rate up to
# a thousand dollars per million tokens against a billion tokens.
RATES = st.integers(min_value=0, max_value=10**6).map(lambda n: Money(nanodollars=n))
COUNTS = st.integers(min_value=0, max_value=10**9)

PRICES = st.builds(
    ModelPrice,
    provider=st.just("anthropic"),
    model=st.just("claude-opus-5"),
    per_input_token=RATES,
    per_cached_input_token=RATES,
    per_output_token=RATES,
)
USAGES = st.builds(
    TokenUsage,
    input_tokens=COUNTS,
    cached_input_tokens=COUNTS,
    output_tokens=COUNTS,
)


@given(PRICES, USAGES)
def test_a_charge_is_never_negative(price: ModelPrice, usage: TokenUsage) -> None:
    assert price.charge_for(usage) >= Money(nanodollars=0)


@given(PRICES, USAGES)
def test_a_charge_is_the_sum_of_what_each_kind_of_token_costs(
    price: ModelPrice, usage: TokenUsage
) -> None:
    parts = (
        price.per_input_token * usage.input_tokens
        + price.per_cached_input_token * usage.cached_input_tokens
        + price.per_output_token * usage.output_tokens
    )
    assert price.charge_for(usage) == parts


@given(PRICES, USAGES, USAGES)
def test_charging_accumulated_usage_is_charging_each_and_adding(
    price: ModelPrice, left: TokenUsage, right: TokenUsage
) -> None:
    # This is what lets a run total be computed either way round: per call as it happens,
    # or once over the accumulated usage. The budget enforcer of phase 4 does both.
    assert price.charge_for(left + right) == price.charge_for(left) + price.charge_for(right)


@given(PRICES, USAGES, COUNTS)
def test_more_tokens_never_cost_less(price: ModelPrice, usage: TokenUsage, extra: int) -> None:
    more = TokenUsage(
        input_tokens=usage.input_tokens + extra,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
    )
    assert price.charge_for(more) >= price.charge_for(usage)


@given(PRICES, COUNTS, COUNTS)
def test_the_reservation_bounds_what_the_call_can_cost(
    price: ModelPrice, input_tokens: int, max_tokens: int
) -> None:
    # A call that respected max_tokens and hit no cache is the worst case the reservation
    # was taken against. If this ever fails, ADR-0004's budgets stop being hard.
    reserved = price.upper_bound(input_tokens=input_tokens, max_tokens=max_tokens)
    for output_tokens in {0, max_tokens // 2, max_tokens}:
        actual = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        assert price.charge_for(actual) <= reserved


@given(PRICES, COUNTS, COUNTS, COUNTS)
def test_a_cached_call_costs_less_than_the_reservation_held_for_it(
    price: ModelPrice, input_tokens: int, cached_input_tokens: int, max_tokens: int
) -> None:
    # The reservation prices every input token at the full rate because it cannot know what
    # the cache will serve. Tokens that do come from the cache are priced no higher than
    # that, so the bound survives them.
    reserved = price.upper_bound(
        input_tokens=input_tokens + cached_input_tokens, max_tokens=max_tokens
    )
    actual = TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=max_tokens,
    )
    if price.per_cached_input_token <= price.per_input_token:
        assert price.charge_for(actual) <= reserved


@given(
    st.decimals(min_value=Decimal(0), max_value=Decimal(1000), places=3, allow_nan=False),
)
def test_a_quoted_rate_converts_to_a_thousand_nanodollars_per_dollar_per_mtok(
    dollars_per_mtok: Decimal,
) -> None:
    # Every rate quoted to three decimal places lands on an exact nanodollar, which is what
    # makes the conversion a one-off and the per-call arithmetic rounding-free.
    price = ModelPrice.from_dollars_per_mtok(
        provider="anthropic",
        model="claude-opus-5",
        input_dollars_per_mtok=dollars_per_mtok,
        cached_input_dollars_per_mtok=0,
        output_dollars_per_mtok=0,
    )
    assert price.per_input_token == Money(nanodollars=int(dollars_per_mtok * 1000))
