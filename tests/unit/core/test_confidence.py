"""Confidence orders by meaning, not by the spelling of its labels.

Three members make exhaustive checking cheaper and stronger than sampling, so every
ordered pair is asserted rather than a random selection of them.
"""

import pytest

from researchmind.core.base import DomainModel
from researchmind.core.confidence import Confidence

ASCENDING = (Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH)


class Sample(DomainModel):
    """A model carrying a confidence, used to check serialisation."""

    level: Confidence


def test_every_ordered_pair_agrees_with_the_declared_order() -> None:
    for i, left in enumerate(ASCENDING):
        for j, right in enumerate(ASCENDING):
            assert (left < right) is (i < j)
            assert (left <= right) is (i <= j)
            assert (left > right) is (i > j)
            assert (left >= right) is (i >= j)


def test_alphabetical_order_would_have_been_wrong() -> None:
    # The whole reason this enum does not inherit from str.
    assert Confidence.HIGH > Confidence.LOW
    assert Confidence.HIGH.value < Confidence.LOW.value


def test_ranks_are_consecutive_from_zero() -> None:
    assert [level.rank for level in ASCENDING] == [0, 1, 2]


def test_comparison_with_another_type_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        _ = Confidence.HIGH < 1


def test_it_renders_and_serialises_as_its_label() -> None:
    assert str(Confidence.HIGH) == "high"
    assert Sample(level=Confidence.HIGH).model_dump_json() == '{"level":"high"}'
    assert Sample.model_validate_json('{"level":"low"}').level is Confidence.LOW
