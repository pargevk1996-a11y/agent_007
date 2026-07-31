"""A statement extracted from a source, and the quote that carries it.

Every field of the binding — the source, the supporting quote, the confidence — is
required. That is what makes the rule in ``executor`` a property of the type rather than a
discipline callers are trusted to keep: a fact without its binding cannot be constructed,
so it cannot be stored, streamed or cited.

``source_id`` is therefore not optional. Design principle 5 allows "no source found", but
that outcome is the absence of a fact, not a fact with a hole in it. It is expressed in the
report as a claim with nothing supporting it, which keeps an unsupported fact
unrepresentable here.

The quote is the text as it appears in the source, and this module stores no offsets into
the document. Containment — is this quote in that document — survives a change in how
documents are parsed and normalised, which character offsets do not, and the parsed
representation does not exist yet. Offsets can be added in phase 9 if disambiguating a
quote that occurs more than once turns out to matter.

``confidence`` is the extractor's own reading of how firmly the quote carries the
statement. It is not a verdict: whether the source supports the claim is the critic's
typed classification (ADR-0006), which arrives separately and may disagree.
"""

from typing import Final

from pydantic import Field

from researchmind.core.base import DomainModel
from researchmind.core.clock import UtcDatetime
from researchmind.core.confidence import Confidence
from researchmind.core.ids import FactId, SourceId, SubQuestionId

MAX_STATEMENT_LENGTH: Final = 1000
"""Longest extracted statement, in characters."""

MAX_QUOTE_LENGTH: Final = 2000
"""Longest supporting quote, in characters.

A bound exists because a quote the size of the document is not a quote — it is a way of
moving the whole text into the critic's context and calling it evidence.
"""


class Fact(DomainModel):
    """One extracted statement, bound to the sub-question that asked for it.

    ``sub_question_id`` is the spine of the trace (ADR-0002): it is how a fact is
    attributed to the step that paid for it. Whether that step and this source exist is an
    invariant across objects, which no single fact can check; it is enforced where facts
    are assembled.
    """

    id: FactId
    sub_question_id: SubQuestionId
    source_id: SourceId
    statement: str = Field(min_length=1, max_length=MAX_STATEMENT_LENGTH)
    quote: str = Field(min_length=1, max_length=MAX_QUOTE_LENGTH)
    confidence: Confidence
    extracted_at: UtcDatetime
