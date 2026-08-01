"""The provider taxonomy: what each failure carries, and whether trying again is sane."""

from datetime import timedelta

import pytest

from researchmind.core.errors import ResearchmindError
from researchmind.providers.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

RETRYABLE: list[type[ProviderError]] = [
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderTimeoutError,
]
NOT_RETRYABLE: list[type[ProviderError]] = [
    ProviderAuthError,
    ProviderRequestError,
    ProviderResponseError,
]


def test_a_provider_error_names_the_provider_and_what_it_said() -> None:
    # The promise made by the root error module, kept here.
    error = ProviderUnavailableError("upstream is down", provider="anthropic", status=503)
    assert error.provider == "anthropic"
    assert error.status == 503
    assert str(error) == "upstream is down"


def test_a_status_is_optional_because_not_every_failure_has_one() -> None:
    # A timeout never produced a response, so there is no code to record.
    assert ProviderTimeoutError("no response in 30s", provider="openai").status is None


@pytest.mark.parametrize("error_type", RETRYABLE)
def test_transient_failures_declare_themselves_retryable(
    error_type: type[ProviderError],
) -> None:
    assert error_type.retryable is True


@pytest.mark.parametrize("error_type", NOT_RETRYABLE)
def test_failures_that_would_repeat_declare_themselves_otherwise(
    error_type: type[ProviderError],
) -> None:
    # Retrying a rejected key or a malformed request reproduces the failure exactly.
    assert error_type.retryable is False


def test_retryability_is_readable_from_the_type_not_only_the_instance() -> None:
    # The retry policy of 3.4 classifies before it catches, so this must not require an
    # instance to consult.
    assert ProviderTimeoutError.retryable is True
    assert ProviderTimeoutError("timed out", provider="anthropic").retryable is True


def test_rate_limiting_carries_the_wait_the_provider_asked_for() -> None:
    error = ProviderRateLimitError(
        "slow down",
        provider="anthropic",
        status=429,
        retry_after=timedelta(seconds=12),
    )
    assert error.retry_after == timedelta(seconds=12)


def test_an_unstated_wait_is_absent_rather_than_zero() -> None:
    # Absent means the provider did not say, which is not the same as "no wait needed".
    assert ProviderRateLimitError("slow down", provider="openai").retry_after is None


@pytest.mark.parametrize("error_type", RETRYABLE + NOT_RETRYABLE)
def test_every_provider_failure_is_catchable_as_one(error_type: type[ProviderError]) -> None:
    with pytest.raises(ProviderError):
        raise error_type("failed", provider="anthropic")


@pytest.mark.parametrize("error_type", RETRYABLE + NOT_RETRYABLE)
def test_every_provider_failure_is_ours(error_type: type[ProviderError]) -> None:
    # `except ResearchmindError` catches everything we consider ours and lets a genuine
    # bug travel upward, which is the whole point of having one root.
    with pytest.raises(ResearchmindError):
        raise error_type("failed", provider="anthropic")


def test_a_provider_failure_is_not_confused_with_an_unrelated_error() -> None:
    assert not issubclass(ProviderError, ValueError)
    assert not issubclass(ProviderError, TypeError)
