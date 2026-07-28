"""Normalisation to UTC preserves the instant and is idempotent, at any offset.

Offsets are generated as fixed deltas rather than named zones: the property under test is
about arithmetic on offsets, and fixed deltas exercise it without depending on a system
timezone database. The date range is bounded away from datetime.min and datetime.max so
that a legitimate conversion cannot overflow — that edge has its own example test.
"""

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from researchmind.core.base import DomainModel
from researchmind.core.clock import UtcDatetime

OFFSETS = st.builds(
    timezone,
    st.timedeltas(min_value=timedelta(hours=-23), max_value=timedelta(hours=23)),
)
AWARE = st.datetimes(
    min_value=datetime(1900, 1, 1),  # noqa: DTZ001  # bounds for the strategy are naive
    max_value=datetime(2200, 1, 1),  # noqa: DTZ001
    timezones=OFFSETS,
)
NAIVE = st.datetimes(
    min_value=datetime(1900, 1, 1),  # noqa: DTZ001
    max_value=datetime(2200, 1, 1),  # noqa: DTZ001
)


class Sample(DomainModel):
    """A model carrying an instant."""

    at: UtcDatetime


@given(AWARE)
def test_normalisation_preserves_the_instant(value: datetime) -> None:
    assert Sample(at=value).at == value


@given(AWARE)
def test_normalisation_is_idempotent(value: datetime) -> None:
    once = Sample(at=value).at
    assert Sample(at=once).at == once


@given(AWARE)
def test_the_stored_offset_is_always_utc(value: datetime) -> None:
    assert Sample(at=value).at.utcoffset() == timedelta(0)


@given(NAIVE)
def test_naive_values_are_always_rejected(value: datetime) -> None:
    with pytest.raises(ValidationError):
        Sample(at=value)
