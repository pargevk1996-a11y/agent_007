"""What Anthropic charges, as of the dates recorded here.

ADR-0004 asks for prices in a versioned file with ``effective_from`` dates. This module is
that file. It is Python rather than TOML or YAML because a Python module is already a
versioned file, is checked by ``mypy --strict``, and needs no loader — and a loader would
need tests of its own to earn nothing, while nothing outside this repository edits these
numbers. If prices ever arrive from somewhere that is not a commit, that is the point to
reconsider, and the ``PriceList`` type will not have to change.

**Provenance.** Rates below are the published Claude API rates for first-party access,
taken from Anthropic's model and pricing documentation and last checked on 2026-08-01. They
are not the rates for Amazon Bedrock or Google Vertex, which are partner-operated and
priced separately; an adapter for either would bring its own history.

**Cache reads are derived, not quoted.** The documentation gives cache reads as
approximately one tenth of the input rate rather than publishing a figure per model, so
``per_cached_input_token`` here is that tenth, computed. Recorded as a derivation so that a
later correction knows what it is correcting.

**There is no rate for cache writes, and that is not an omission.** Writes are billed above
the input rate — 1.25x at the five-minute lifetime — while the adapter folds cache-creation
tokens into ``input_tokens`` and so prices them at the ordinary rate. The saving that
``TokenUsage`` splits out is the read discount, which is measured exactly; the write premium
is under-reported by design, which errs low on cost rather than high on a discount. Adding a
fourth rate here would not fix it: the counts arrive already folded together.

**Two lists, because one would be wrong on one side of a date.** ``claude-sonnet-5`` runs an
introductory rate that ends on 2026-08-31. Holding only the standard rate would over-state
every Sonnet cost recorded in August, and a cost is supposed to be what was charged; holding
only the introductory rate would under-reserve budget from September. Both are recorded, and
``PriceListHistory`` decides which one a given instant falls under.

The provider name is imported from the adapter rather than written out again here. It costs
this module an import of the vendor SDK, and buys the guarantee that the string identifying
these prices and the string recorded on every cost row are the same string.

Models we could not use are absent rather than listed for completeness: ``claude-fable-5``
requires a data-retention configuration we have not agreed to, and ``claude-mythos-5`` needs
a programme we are not in. An unpriced model raises, so their absence is enforced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from researchmind.providers.anthropic_adapter import PROVIDER_NAME
from researchmind.providers.pricing import ModelPrice, PriceList, PriceListHistory


def _anthropic(model: str, *, dollars_in: str, dollars_cached: str, dollars_out: str) -> ModelPrice:
    """Build one Anthropic price from the published per-million-token dollars."""
    return ModelPrice.from_dollars_per_mtok(
        provider=PROVIDER_NAME,
        model=model,
        input_dollars_per_mtok=dollars_in,
        cached_input_dollars_per_mtok=dollars_cached,
        output_dollars_per_mtok=dollars_out,
    )


_OPUS_5: Final = _anthropic(
    "claude-opus-5", dollars_in="5.00", dollars_cached="0.50", dollars_out="25.00"
)
_HAIKU_4_5: Final = _anthropic(
    "claude-haiku-4-5", dollars_in="1.00", dollars_cached="0.10", dollars_out="5.00"
)
_SONNET_5_INTRODUCTORY: Final = _anthropic(
    "claude-sonnet-5", dollars_in="2.00", dollars_cached="0.20", dollars_out="10.00"
)
_SONNET_5_STANDARD: Final = _anthropic(
    "claude-sonnet-5", dollars_in="3.00", dollars_cached="0.30", dollars_out="15.00"
)

ANTHROPIC_2026_08: Final = PriceList(
    version="anthropic-2026-08",
    effective_from=datetime(2026, 8, 1, tzinfo=UTC),
    prices=(_OPUS_5, _SONNET_5_INTRODUCTORY, _HAIKU_4_5),
)
"""Current prices, with Sonnet 5 at its introductory rate through 2026-08-31."""

ANTHROPIC_2026_09: Final = PriceList(
    version="anthropic-2026-09",
    effective_from=datetime(2026, 9, 1, tzinfo=UTC),
    prices=(_OPUS_5, _SONNET_5_STANDARD, _HAIKU_4_5),
)
"""Prices from 2026-09-01, when Sonnet 5's introductory rate ends."""

ANTHROPIC_PRICES: Final = PriceListHistory(
    provider=PROVIDER_NAME,
    lists=(ANTHROPIC_2026_08, ANTHROPIC_2026_09),
)
"""Every set of Anthropic prices this repository knows, oldest first.

Nothing before 2026-08-01 is priced. This project has no calls from before then, and
inventing a rate for a period we did not observe would produce a number nobody could check.
"""
