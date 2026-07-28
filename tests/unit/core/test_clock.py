"""A naive datetime is refused; an aware one at any offset is normalised."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from researchmind.core.base import DomainModel
from researchmind.core.clock import UtcDatetime, utc_now


class Sample(DomainModel):
    """A model carrying an instant."""

    at: UtcDatetime


def test_a_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(at=datetime(2026, 7, 28, 12, 0, 0))  # noqa: DTZ001  # the point of the test


def test_another_offset_is_accepted_and_normalised() -> None:
    moscow = timezone(timedelta(hours=3))
    sample = Sample(at=datetime(2026, 7, 28, 15, 0, 0, tzinfo=moscow))
    assert sample.at == datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
    assert sample.at.tzinfo == UTC


def test_an_instant_that_cannot_be_expressed_in_utc_fails_validation() -> None:
    # datetime.max at a western offset overflows on conversion. It should arrive as a
    # validation error, not as an OverflowError from inside the parser.
    unrepresentable = datetime.max.replace(tzinfo=timezone(timedelta(hours=-14)))
    with pytest.raises(ValidationError):
        Sample(at=unrepresentable)


def test_utc_now_is_aware_and_in_utc() -> None:
    now = utc_now()
    assert now.tzinfo is UTC
    assert now.utcoffset() == timedelta(0)
