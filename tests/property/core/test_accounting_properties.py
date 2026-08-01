"""Properties of the accounting types, over generated counts and limits.

Token addition is arithmetic, and arithmetic is worth checking algebraically rather than
on three worked examples. The other three properties turn a stated invariant into an
exhaustive claim: over every combination of limits, over every kind of call, over every
amount.
"""

from datetime import timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from researchmind.core.budget import Budget, BudgetScopeKind
from researchmind.core.clock import utc_now
from researchmind.core.cost import CallKind, Cost
from researchmind.core.ids import new_call_id, new_run_id
from researchmind.core.money import Money
from researchmind.core.tokens import TokenUsage

COUNTS = st.integers(min_value=0, max_value=10**12)
NANODOLLARS = st.integers(min_value=-(10**15), max_value=10**15)
SCOPES = st.sampled_from(list(BudgetScopeKind))
KINDS = st.sampled_from(list(CallKind))

USAGES = st.builds(
    TokenUsage,
    input_tokens=COUNTS,
    cached_input_tokens=COUNTS,
    output_tokens=COUNTS,
)

TOKEN_LIMITS = st.one_of(st.none(), st.integers(min_value=1, max_value=10**9))
SPEND_LIMITS = st.one_of(
    st.none(), st.integers(min_value=1, max_value=10**12).map(lambda n: Money(nanodollars=n))
)
DURATION_LIMITS = st.one_of(
    st.none(), st.integers(min_value=1, max_value=10**6).map(lambda s: timedelta(seconds=s))
)


def _cost(kind: CallKind, tokens: TokenUsage | None, amount: Money) -> Cost:
    return Cost(
        call_id=new_call_id(),
        run_id=new_run_id(),
        kind=kind,
        provider="anthropic",
        model="claude-opus-5",
        tokens=tokens,
        amount=amount,
        latency=timedelta(milliseconds=1),
        price_version="generated",
        incurred_at=utc_now(),
    )


@given(USAGES)
def test_the_total_is_the_sum_of_the_parts(usage: TokenUsage) -> None:
    assert usage.total == usage.input_tokens + usage.cached_input_tokens + usage.output_tokens


@given(USAGES, USAGES)
def test_accumulation_commutes(left: TokenUsage, right: TokenUsage) -> None:
    assert left + right == right + left


@given(USAGES, USAGES, USAGES)
def test_accumulation_associates(a: TokenUsage, b: TokenUsage, c: TokenUsage) -> None:
    assert (a + b) + c == a + (b + c)


@given(USAGES, USAGES)
def test_totals_add_up_as_the_usages_do(left: TokenUsage, right: TokenUsage) -> None:
    assert (left + right).total == left.total + right.total


@given(SCOPES, TOKEN_LIMITS, SPEND_LIMITS, DURATION_LIMITS)
def test_a_budget_is_accepted_exactly_when_it_bounds_something(
    scope: BudgetScopeKind,
    max_tokens: int | None,
    max_spend: Money | None,
    max_duration: timedelta | None,
) -> None:
    bounds_something = any(limit is not None for limit in (max_tokens, max_spend, max_duration))

    if bounds_something:
        budget = Budget(
            scope=scope,
            max_tokens=max_tokens,
            max_spend=max_spend,
            max_duration=max_duration,
        )
        assert budget.scope is scope
    else:
        with pytest.raises(ValidationError):
            Budget(
                scope=scope,
                max_tokens=max_tokens,
                max_spend=max_spend,
                max_duration=max_duration,
            )


@given(KINDS, st.one_of(st.none(), USAGES))
def test_tokens_are_recorded_exactly_when_the_call_spends_them(
    kind: CallKind, tokens: TokenUsage | None
) -> None:
    spends_tokens = kind is not CallKind.TOOL
    coherent = spends_tokens is (tokens is not None)

    if coherent:
        assert _cost(kind, tokens, Money(nanodollars=1)).kind is kind
    else:
        with pytest.raises(ValidationError):
            _cost(kind, tokens, Money(nanodollars=1))


@given(NANODOLLARS)
def test_a_cost_is_accepted_exactly_when_it_is_not_negative(nanodollars: int) -> None:
    amount = Money(nanodollars=nanodollars)

    if nanodollars >= 0:
        assert _cost(CallKind.TOOL, None, amount).amount == amount
    else:
        with pytest.raises(ValidationError):
            _cost(CallKind.TOOL, None, amount)
