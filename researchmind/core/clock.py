"""Time, with the ambiguity removed.

Every instant the system records is timezone-aware and expressed in UTC. A naive datetime
is rejected rather than assumed to be local or assumed to be UTC, because both
assumptions are wrong somewhere and neither announces itself.

An aware value at any other offset is accepted and normalised. A client sending
``+03:00`` is not making a mistake; a client sending no offset at all is.

The module is named ``clock`` rather than ``time`` so that it does not shadow the
standard library module of that name.
"""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator


def _require_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalise aware ones to UTC.

    Normalisation is idempotent and preserves the instant, so validating an already
    validated value is a no-op.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        msg = "datetime must be timezone-aware; a naive value has no defined instant"
        raise ValueError(msg)
    try:
        return value.astimezone(UTC)
    except (OverflowError, OSError) as exc:
        # datetime.max at a negative offset cannot be expressed in UTC. Surfacing it as a
        # ValueError keeps it inside pydantic's validation report instead of escaping as
        # an OverflowError from somewhere unrelated.
        msg = "datetime cannot be represented in UTC"
        raise ValueError(msg) from exc


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
"""A timezone-aware instant, normalised to UTC. Use this, never a bare ``datetime``."""


def utc_now() -> datetime:
    """Return the current instant, timezone-aware and in UTC."""
    return datetime.now(UTC)
