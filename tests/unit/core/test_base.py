"""The guarantees DomainModel makes to every type built on it."""

import pytest
from pydantic import ValidationError

from researchmind.core.base import DomainModel


class Sample(DomainModel):
    """A throwaway model used to exercise the base configuration."""

    name: str
    count: int


def test_attributes_cannot_be_reassigned() -> None:
    sample = Sample(name="a", count=1)
    with pytest.raises(ValidationError):
        # Suppressed because mypy already refuses this assignment; the test asserts
        # that the runtime refuses it too, for callers that are not type-checked.
        sample.name = "b"  # type: ignore[misc]


def test_unexpected_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample.model_validate({"name": "a", "count": 1, "hallucinated": True})


def test_surrounding_whitespace_is_stripped() -> None:
    assert Sample(name="  spaced  ", count=1).name == "spaced"


def test_models_are_hashable_and_compare_by_value() -> None:
    first = Sample(name="a", count=1)
    second = Sample(name="a", count=1)
    assert first == second
    assert len({first, second}) == 1
