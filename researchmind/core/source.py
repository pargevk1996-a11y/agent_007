"""A document we actually retrieved, pinned to what it said when we read it.

A source is not a URL. A URL is an address that may serve different bytes tomorrow; a
source is the address together with the instant we read it and a digest of what we got.
Evidence checked against one set of bytes is not evidence about another, and the replay
mode of ADR-0005 is only honest if the document being replayed is provably the one that
was recorded.

Two dates, deliberately not one. ``retrieved_at`` is ours and always known.
``published_at`` belongs to the document, is frequently absent, and is therefore optional.
Collapsing them would let "we fetched this today" quietly read as "this is from today",
which is the specific lie a research tool must not tell.

The URL is stored exactly as it was fetched. Pydantic's ``HttpUrl`` would normalise it —
a slash appended to a bare host, punycode and percent-encoding rewritten — and a citation
that differs from the address we requested is a small untruth in a system whose entire
value is that its claims can be checked. The scheme and the host are validated; nothing is
rewritten.

Deduplication is not a concern of this type. The same URL fetched twice is legitimately
two sources with two digests; whether storage collapses them is storage's decision.
"""

from typing import Annotated, Final
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field

from researchmind.core.base import DomainModel
from researchmind.core.clock import UtcDatetime
from researchmind.core.ids import SourceId

MAX_URL_LENGTH: Final = 2048
"""Longest URL accepted. The figure is the long-standing practical ceiling."""

MAX_TITLE_LENGTH: Final = 500
"""Longest source title, in characters."""

SHA256_HEX_PATTERN: Final = r"^[0-9a-f]{64}$"
"""A SHA-256 digest as ``hexdigest`` writes it: sixty-four lower-case hex characters."""

_FETCHABLE_SCHEMES: Final = frozenset({"http", "https"})


def _require_fetchable_url(value: str) -> str:
    """Accept an http or https URL with a host, and return it unchanged.

    Unchanged is the point. Validation here answers "could this have been fetched, and can
    a reader follow it", not "what is the canonical form of this address".
    """
    if any(character.isspace() for character in value):
        # Surrounding whitespace is already stripped by DomainModel, so anything left is
        # interior: a line break from a PDF, or two URLs that were run together.
        msg = "a URL cannot contain whitespace"
        raise ValueError(msg)
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        msg = "URL cannot be parsed"
        raise ValueError(msg) from exc
    if parts.scheme not in _FETCHABLE_SCHEMES:
        msg = f"URL scheme must be http or https, not {parts.scheme!r}"
        raise ValueError(msg)
    if not parts.hostname:
        msg = "URL names no host"
        raise ValueError(msg)
    return value


FetchedUrl = Annotated[
    str,
    Field(min_length=1, max_length=MAX_URL_LENGTH),
    AfterValidator(_require_fetchable_url),
]
"""An http or https URL, stored byte for byte as it was fetched."""

Sha256Hex = Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
"""A SHA-256 digest in hexadecimal."""


class Source(DomainModel):
    """A retrieved document, identified by its address and its contents.

    ``content_sha256`` binds the source to an exact byte sequence. This type checks the
    shape of the digest and can check nothing else: that it was computed over the text
    that was actually parsed is a contract of the fetcher in phase 5.
    """

    id: SourceId
    url: FetchedUrl
    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    retrieved_at: UtcDatetime
    published_at: UtcDatetime | None = None
    content_sha256: Sha256Hex
