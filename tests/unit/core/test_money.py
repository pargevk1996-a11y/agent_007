"""Worked examples for Money, including the amounts this system actually deals in."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from researchmind.core.money import Money


def test_a_per_token_price_survives_the_round_trip() -> None:
    # Three dollars per million tokens is three millionths of a dollar per token, which
    # is the order of magnitude that makes floating point unusable here.
    price = Money.from_dollars(Decimal("0.000003"))
    assert price.nanodollars == 3000
    assert price.as_dollars() == Decimal("0.000003")


def test_dollars_accept_decimal_int_and_str() -> None:
    assert Money.from_dollars(2).nanodollars == 2_000_000_000
    assert Money.from_dollars("2.5").nanodollars == 2_500_000_000
    assert Money.from_dollars(Decimal("2.5")).nanodollars == 2_500_000_000


def test_fractions_below_a_nanodollar_round_half_to_even() -> None:
    assert Money.from_dollars(Decimal("0.0000000005")).nanodollars == 0
    assert Money.from_dollars(Decimal("0.0000000015")).nanodollars == 2


def test_arithmetic_stays_exact() -> None:
    tenth = Money.from_dollars("0.1")
    assert tenth + tenth + tenth == Money.from_dollars("0.3")
    assert Money.from_dollars("1") - tenth == Money.from_dollars("0.9")
    assert Money.from_dollars("0.000003") * 1000 == Money.from_dollars("0.003")


def test_negative_amounts_are_allowed() -> None:
    # Releasing a budget reservation is a negative movement (ADR-0004).
    assert (Money.from_dollars("1") - Money.from_dollars("3")).nanodollars == -2_000_000_000


def test_amounts_order_by_value() -> None:
    assert Money.from_dollars("0.000003") < Money.from_dollars("0.000004")
    assert Money.from_dollars("1") >= Money.from_dollars("1")


@pytest.mark.parametrize(
    ("nanodollars", "rendered"),
    [(0, "$0"), (3000, "$0.000003"), (-3000, "$-0.000003"), (1_000_000_000_000, "$1000")],
)
def test_rendering_drops_noise_without_resorting_to_exponents(
    nanodollars: int, rendered: str
) -> None:
    assert str(Money(nanodollars=nanodollars)) == rendered


def test_amounts_outside_the_storable_range_are_rejected() -> None:
    # The persisted column is a bigint; an amount that cannot be stored should fail
    # where it is built rather than at INSERT.
    with pytest.raises(ValidationError):
        Money(nanodollars=2**63)
