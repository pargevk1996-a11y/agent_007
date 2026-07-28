"""Algebraic properties of Money, checked over generated amounts rather than examples.

Bounds are chosen so that intermediate results stay inside the storable range: a million
dollars is 1e15 nanodollars, and the largest expression here multiplies that by a
thousand, which is still two orders below the bigint ceiling.
"""

from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from researchmind.core.money import Money

DOLLARS = st.decimals(
    min_value=Decimal("-1000000"),
    max_value=Decimal("1000000"),
    places=9,
    allow_nan=False,
    allow_infinity=False,
)
AMOUNTS = DOLLARS.map(Money.from_dollars)
FACTORS = st.integers(min_value=-1000, max_value=1000)


@given(DOLLARS)
def test_dollars_survive_the_round_trip(amount: Decimal) -> None:
    assert Money.from_dollars(amount).as_dollars() == amount


@given(AMOUNTS, AMOUNTS)
def test_addition_commutes(left: Money, right: Money) -> None:
    assert left + right == right + left


@given(AMOUNTS, AMOUNTS, AMOUNTS)
def test_addition_associates(a: Money, b: Money, c: Money) -> None:
    assert (a + b) + c == a + (b + c)


@given(AMOUNTS, AMOUNTS)
def test_subtraction_undoes_addition(left: Money, right: Money) -> None:
    assert (left + right) - right == left


@given(AMOUNTS, FACTORS, FACTORS)
def test_multiplication_distributes_over_the_factor(amount: Money, first: int, second: int) -> None:
    assert amount * (first + second) == amount * first + amount * second


@given(AMOUNTS, AMOUNTS)
def test_ordering_agrees_with_the_underlying_amount(left: Money, right: Money) -> None:
    assert (left < right) is (left.nanodollars < right.nanodollars)
    assert (left <= right) is (not right < left)


@given(AMOUNTS)
def test_the_rendered_form_parses_back_to_the_same_amount(amount: Money) -> None:
    assert Money.from_dollars(str(amount).removeprefix("$")) == amount
