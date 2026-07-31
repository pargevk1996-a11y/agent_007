"""Properties of sources and facts, checked over generated evidence.

Three claims are worth generating rather than illustrating: that any real digest is
accepted and anything shaped differently is not, that a URL survives validation unchanged,
and that the whitespace a document arrives with never decides whether two quotes are the
same.
"""

from datetime import datetime
from hashlib import sha256

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.confidence import Confidence
from researchmind.core.fact import Fact
from researchmind.core.ids import (
    FactId,
    SourceId,
    SubQuestionId,
    new_fact_id,
    new_source_id,
    new_sub_question_id,
)
from researchmind.core.source import Source

SCHEMES = st.sampled_from(["http", "https"])
HOSTS = st.from_regex(r"[a-z][a-z0-9-]{0,20}(\.[a-z]{2,6}){1,2}", fullmatch=True)
PATHS = st.from_regex(r"(/[A-Za-z0-9._~%-]{0,20}){0,3}", fullmatch=True)
QUERIES = st.one_of(st.just(""), st.from_regex(r"\?[A-Za-z0-9=&._-]{1,30}", fullmatch=True))

CORE_QUOTES = st.text(min_size=1, max_size=200).map(str.strip).filter(bool)
PADDING = st.text(alphabet=" \t\n\r", max_size=5)


def _is_hex_digest(candidate: str) -> bool:
    stripped = candidate.strip()
    return len(stripped) == 64 and all(character in "0123456789abcdef" for character in stripped)


@st.composite
def urls(draw: st.DrawFn) -> str:
    """Assemble a URL from parts that are all legal unencoded."""
    return f"{draw(SCHEMES)}://{draw(HOSTS)}{draw(PATHS)}{draw(QUERIES)}"


def _source(url: str, digest: str) -> Source:
    return Source(
        id=new_source_id(),
        url=url,
        title="A generated document",
        retrieved_at=utc_now(),
        content_sha256=digest,
    )


def _fact(
    quote: str,
    fact_id: FactId,
    sub_question_id: SubQuestionId,
    source_id: SourceId,
    extracted_at: datetime,
) -> Fact:
    # Every field but the quote is passed in, so that two facts differ in the quote alone
    # and equality is a statement about the quote rather than about the clock.
    return Fact(
        id=fact_id,
        sub_question_id=sub_question_id,
        source_id=source_id,
        statement="A generated statement.",
        quote=quote,
        confidence=Confidence.MEDIUM,
        extracted_at=extracted_at,
    )


@given(st.binary(max_size=2000))
def test_any_real_digest_is_accepted_and_kept(payload: bytes) -> None:
    digest = sha256(payload).hexdigest()
    assert _source("https://example.org/doc", digest).content_sha256 == digest


@given(st.binary(max_size=2000))
def test_an_upper_case_digest_is_rejected(payload: bytes) -> None:
    digest = sha256(payload).hexdigest()
    # A digest of nothing but decimal characters is unchanged by upper(); it is
    # astronomically unlikely and would not be a counterexample to anything.
    assume(digest != digest.upper())
    with pytest.raises(ValidationError):
        _source("https://example.org/doc", digest.upper())


@given(st.text(max_size=80).filter(lambda candidate: not _is_hex_digest(candidate)))
def test_anything_that_is_not_a_hexdigest_is_rejected(candidate: str) -> None:
    with pytest.raises(ValidationError):
        _source("https://example.org/doc", candidate)


@given(urls())
def test_a_valid_url_survives_validation_byte_for_byte(url: str) -> None:
    # The guarantee behind citing an address rather than a canonicalised version of it.
    assert _source(url, sha256(b"").hexdigest()).url == url


@given(CORE_QUOTES, PADDING, PADDING, PADDING, PADDING)
def test_ragged_edges_never_decide_whether_two_quotes_are_the_same(
    quote: str,
    left: str,
    right: str,
    other_left: str,
    other_right: str,
) -> None:
    # Quotes are cut out of documents and arrive with whatever whitespace surrounded them.
    # Two extractions of the same span must be one fact, including as a set member.
    fact_id, sub_question_id, source_id = new_fact_id(), new_sub_question_id(), new_source_id()
    at = utc_now()
    first = _fact(f"{left}{quote}{right}", fact_id, sub_question_id, source_id, at)
    second = _fact(f"{other_left}{quote}{other_right}", fact_id, sub_question_id, source_id, at)

    assert first.quote == quote
    assert first == second
    assert len({first, second}) == 1
