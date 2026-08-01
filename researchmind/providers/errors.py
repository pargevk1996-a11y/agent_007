"""Everything a provider can do to us, as types rather than as status codes.

The root of the project's error tree promises that "a provider error carries the provider
and the status". This module keeps that promise. A caller catching ``ProviderError`` knows
which vendor failed and what it said, without parsing a message.

The taxonomy exists to answer one question the retry policy of phase 3.4 will ask
constantly: is trying again a reasonable thing to do? That answer belongs here, on the
type, and not in a table somewhere else that has to be kept in step with it. A timeout is
worth retrying and a rejected key is not, and neither fact depends on who is asking.

The split is by *what to do about it*, not by HTTP status. Two providers return different
codes for the same condition, and one of them will change its mind in a minor version.
"""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from researchmind.core.errors import ResearchmindError


class ProviderError(ResearchmindError):
    """Base class for every failure originating with an LLM or embedding provider.

    ``retryable`` is a property of the failure, not of the caller's patience. Subclasses
    set it once; the retry policy reads it and adds the backoff, the ceiling and the
    budget check, none of which belong here.
    """

    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, provider: str, status: int | None = None) -> None:
        """Record which provider failed, and what it reported if it reported anything."""
        super().__init__(message)
        self.provider = provider
        self.status = status


class ProviderAuthError(ProviderError):
    """Credentials were rejected. Retrying with the same key changes nothing."""


class ProviderRequestError(ProviderError):
    """The provider refused the request as malformed or over a hard limit.

    This is our defect: a context longer than the model accepts, a schema it will not
    take, a parameter outside its range. Retrying identical input reproduces it exactly.
    """


class ProviderResponseError(ProviderError):
    """The call succeeded and the body could not be understood.

    Distinct from a refusal, because the failure is on the way back. Whether a retry helps
    depends on why, so this does not claim it does.
    """


class ProviderRateLimitError(ProviderError):
    """The provider is throttling us.

    ``retry_after`` carries the wait the provider asked for, when it named one. Absent
    means it did not say, not that waiting is unnecessary.
    """

    retryable: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status: int | None = None,
        retry_after: timedelta | None = None,
    ) -> None:
        """Record the requested wait alongside the provider and status."""
        super().__init__(message, provider=provider, status=status)
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderError):
    """The provider is failing on its side. The same request may well succeed later."""

    retryable: ClassVar[bool] = True


class ProviderTimeoutError(ProviderError):
    """No response arrived inside the deadline.

    Retryable, with the caveat that the original call may still be running and may still
    be billed. Cost accounting cannot see what it was never told about.
    """

    retryable: ClassVar[bool] = True
