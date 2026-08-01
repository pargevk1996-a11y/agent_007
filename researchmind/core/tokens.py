"""Token counts as a value, and the arithmetic of adding them up.

Usage is reported per call and consumed per scope: the budget enforcer of phase 4 needs
the total for a sub-question and for a run, which is the same operation applied twice.
That addition is pure arithmetic over three integers, so it belongs beside ``Money`` in
the domain rather than being written out again in whichever module happens to need it.

Cached input tokens are counted separately because they are priced separately. ADR-0003
keeps the provider interface narrow specifically so that vendor features worth having
survive it, and names Anthropic's prompt caching as one of them — a large saving on the
long source excerpts this system sends. An accounting type that folds cached tokens into
ordinary input tokens cannot measure that saving, which makes the saving unarguable in
either direction.

One honest limitation: zero means "reported as zero". Not every provider reports cached
tokens at all, and an adapter facing one that does not will write zero here. Distinguishing
"none were cached" from "nobody said" is a job for the trace, not for a second layer of
optionality threaded through every consumer of this type.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from researchmind.core.base import DomainModel

# The persisted columns are bigints, for the same reason Money is bounded: a value that
# could be constructed but never stored should fail where it is built.
_MAX_TOKENS: Final = 2**63 - 1


class TokenUsage(DomainModel):
    """What one call consumed, split by how the tokens were priced."""

    input_tokens: int = Field(ge=0, le=_MAX_TOKENS)
    cached_input_tokens: int = Field(default=0, ge=0, le=_MAX_TOKENS)
    output_tokens: int = Field(ge=0, le=_MAX_TOKENS)

    @property
    def total(self) -> int:
        """Return every token the call touched, however it was priced."""
        return self.input_tokens + self.cached_input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Accumulate usage across calls, keeping the split intact."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )
