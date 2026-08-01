"""Token counts, their total and their accumulation."""

import pytest
from pydantic import ValidationError

from researchmind.core.tokens import TokenUsage


def test_the_total_counts_every_token_however_it_was_priced() -> None:
    usage = TokenUsage(input_tokens=100, cached_input_tokens=900, output_tokens=50)
    assert usage.total == 1050


def test_cached_tokens_default_to_none_counted() -> None:
    # The default is for adapters facing a provider that reports no cache at all. Zero
    # here means "reported as zero", which is the limitation stated in the module.
    assert TokenUsage(input_tokens=10, output_tokens=5).cached_input_tokens == 0


def test_usage_accumulates_across_calls_keeping_the_split() -> None:
    first = TokenUsage(input_tokens=10, cached_input_tokens=100, output_tokens=5)
    second = TokenUsage(input_tokens=20, cached_input_tokens=200, output_tokens=7)
    combined = first + second
    assert combined.input_tokens == 30
    assert combined.cached_input_tokens == 300
    assert combined.output_tokens == 12
    assert combined.total == first.total + second.total


@pytest.mark.parametrize(
    ("field", "value"),
    [("input_tokens", -1), ("cached_input_tokens", -1), ("output_tokens", -1)],
)
def test_a_negative_count_is_rejected(field: str, value: int) -> None:
    payload: dict[str, int] = {"input_tokens": 1, "output_tokens": 1, field: value}
    with pytest.raises(ValidationError):
        TokenUsage.model_validate(payload)


def test_a_count_outside_the_storable_range_is_rejected() -> None:
    # The persisted column is a bigint, so an unstorable count fails where it is built.
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=2**63, output_tokens=0)


def test_usage_compares_by_value() -> None:
    first = TokenUsage(input_tokens=1, output_tokens=2)
    second = TokenUsage(input_tokens=1, cached_input_tokens=0, output_tokens=2)
    assert first == second
    assert len({first, second}) == 1
