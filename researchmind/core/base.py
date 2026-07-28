"""The base class every domain model inherits.

Configuration lives here rather than being repeated per model, so the rules that make a
domain object trustworthy are stated once and cannot quietly differ between types.
"""

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """A frozen, closed value object.

    Three settings carry weight:

    - ``frozen`` — the system is built on immutable records (ADR-0001, ADR-0002). A
      change produces a new object; nothing is edited in place. Freezing also makes
      models hashable, which lets them be used as dictionary keys and set members.
    - ``extra="forbid"`` — an unexpected field in a payload is a schema violation, not a
      bonus. This matters most where a model parses LLM output: a hallucinated key must
      fail loudly rather than be dropped in silence.
    - ``str_strip_whitespace`` — quoted spans and URLs arrive from documents with ragged
      edges, and trailing whitespace should never be the reason two of them differ.

    Freezing protects attributes, not containers. Collection fields therefore use
    ``tuple`` rather than ``list`` throughout the domain.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )
