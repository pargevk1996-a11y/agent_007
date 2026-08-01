"""Everything a provider can do to us, as types rather than as status codes.

The root of the project's error tree promises that "a provider error carries the provider
and the status". This module keeps that promise. A caller catching ``ProviderError`` knows
which vendor failed and what it said, without parsing a message.

The taxonomy exists to answer one question the retry policy of a later increment will ask
constantly: is trying again a reasonable thing to do? That answer belongs here, on the
type, and not in a table somewhere else that has to be kept in step with it. A timeout is
worth retrying and a rejected key is not, and neither fact depends on who is asking.

The split is by *what to do about it*, not by HTTP status. Two providers return different
codes for the same condition, and one of them will change its mind in a minor version.

One member of the tree is ours rather than the vendor's — ``UnknownModelPriceError``,
raised while accounting for a call that did succeed. It is here so that code guarding a
provider call with ``except ProviderError`` catches everything that call can raise.
"""

from __future__ import annotations

from datetime import datetime, timedelta
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


class UnknownModelPriceError(ProviderError):
    """The configured price list does not price the model that served a call.

    Ours rather than the vendor's, and therefore not retryable: the same call priced
    against the same list fails the same way. It lives in the provider subtree even so,
    because it is raised while accounting for a provider call, and a caller that guards a
    call with ``except ProviderError`` should not have a second root escaping past it.
    """

    def __init__(self, *, provider: str, model: str, price_version: str) -> None:
        """Record which model went unpriced, and which price list failed to price it."""
        super().__init__(
            f"price list {price_version!r} does not price model {model!r} from {provider!r}",
            provider=provider,
        )
        self.model = model
        self.price_version = price_version


class UnpricedInstantError(ProviderError):
    """No price list was in effect when the call happened.

    Raised when an instant falls before the earliest prices we hold. Returning the oldest
    list instead would price a call at rates that had not been announced when it was made,
    and would do so silently — the cost row would carry a version that disagrees with its
    own timestamp, which is precisely what ``price_version`` exists to make checkable.
    """

    def __init__(self, *, provider: str, instant: datetime) -> None:
        """Record which provider, and the instant nothing covers."""
        super().__init__(
            f"no {provider!r} price list was in effect at {instant.isoformat()}",
            provider=provider,
        )
        self.instant = instant


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
