"""What a source must carry, and the addresses it refuses."""

from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256

import pytest
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.ids import new_source_id
from researchmind.core.source import MAX_TITLE_LENGTH, MAX_URL_LENGTH, Source

DIGEST = sha256(b"the document as it was parsed").hexdigest()


def _source(
    *,
    url: str = "https://example.org/stablecoins-2025",
    title: str = "Stablecoin regulation in 2025",
    retrieved_at: datetime | None = None,
    published_at: datetime | None = None,
    content_sha256: str = DIGEST,
) -> Source:
    return Source(
        id=new_source_id(),
        url=url,
        title=title,
        retrieved_at=retrieved_at if retrieved_at is not None else utc_now(),
        published_at=published_at,
        content_sha256=content_sha256,
    )


def test_a_source_carries_its_address_and_its_digest() -> None:
    source = _source()
    assert source.url == "https://example.org/stablecoins-2025"
    assert source.content_sha256 == DIGEST
    assert source.published_at is None


def test_the_url_is_stored_exactly_as_fetched() -> None:
    # A bare host keeps its lack of a trailing slash, and encodings are not rewritten.
    # What is cited must be what was requested.
    assert _source(url="https://example.org").url == "https://example.org"
    assert (
        _source(url="https://example.org/a%20b?q=1&r=2").url == "https://example.org/a%20b?q=1&r=2"
    )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org/paper.pdf",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "example.org/paper",
        "/relative/path",
        "http:///no-host",
        "https://exa mple.org/paper",
        "https://example.org/a\nb",
        "",
    ],
)
def test_an_address_we_could_not_have_fetched_is_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        _source(url=url)


def test_a_url_past_the_length_ceiling_is_rejected() -> None:
    prefix = "https://example.org/"
    assert len(_source(url=prefix + "x" * (MAX_URL_LENGTH - len(prefix))).url) == MAX_URL_LENGTH
    with pytest.raises(ValidationError):
        _source(url=prefix + "x" * (MAX_URL_LENGTH - len(prefix) + 1))


@pytest.mark.parametrize("title", ["", "   "])
def test_a_source_without_a_title_is_rejected(title: str) -> None:
    # A citation with no title is a bare link, which is not a citation a reader can weigh.
    with pytest.raises(ValidationError):
        _source(title=title)


def test_a_title_past_the_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _source(title="t" * (MAX_TITLE_LENGTH + 1))


@pytest.mark.parametrize(
    "digest",
    [
        DIGEST.upper(),
        DIGEST[:-1],
        DIGEST + "0",
        "z" * 64,
        "",
    ],
)
def test_a_digest_that_is_not_a_sha256_hexdigest_is_rejected(digest: str) -> None:
    with pytest.raises(ValidationError):
        _source(content_sha256=digest)


def test_the_retrieval_instant_must_be_aware() -> None:
    with pytest.raises(ValidationError):
        _source(retrieved_at=datetime(2026, 7, 28, 12, 0))  # noqa: DTZ001


def test_a_publication_date_is_optional_but_must_be_aware_when_given() -> None:
    published = datetime(2025, 3, 1, 9, 0, tzinfo=timezone(timedelta(hours=3)))
    assert _source(published_at=published).published_at is not None
    with pytest.raises(ValidationError):
        _source(published_at=datetime(2025, 3, 1, 9, 0))  # noqa: DTZ001


def test_publication_and_retrieval_are_not_the_same_date() -> None:
    # The distinction is the point: fetching an old document today does not make it new.
    published = datetime(2020, 1, 1, tzinfo=UTC)
    source = _source(published_at=published)
    assert source.published_at is not None
    assert source.published_at < source.retrieved_at


def test_the_same_url_fetched_twice_gives_two_sources() -> None:
    # Two retrievals are two documents whose contents may differ. Collapsing them is a
    # storage decision, not an invariant of the domain.
    first = _source(content_sha256=sha256(b"monday").hexdigest())
    second = _source(content_sha256=sha256(b"tuesday").hexdigest())
    assert first.url == second.url
    assert first != second
