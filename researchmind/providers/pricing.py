"""What a call costs, computed from its usage against a named version of the prices.

ADR-0004 requires a cost to be stored as an absolute number that never moves when a vendor
changes its price list, and ``core.cost.Cost`` records a ``price_version`` so that the
number stays checkable. This module is what makes that version mean something: a
``PriceList`` names itself, and only a price list can build a cost row, so the version
stamped on the row and the numbers inside it cannot disagree.

Prices are held **per token**, not per million tokens, although vendors quote the latter
and ``from_dollars_per_mtok`` takes it in that form. The reason is exactness. ``Money``
multiplies by a whole number, so a per-token price times a token count is an exact integer
operation with no rounding at all, at any call volume. Storing the quoted per-million price
instead would mean a division per call and therefore a rounding decision on every cost row
in the system, repeated for every consumer that ever recomputes one. The rounding happens
once, here, when the price is constructed, and the resolution is generous: a nanodollar per
token is a thousandth of a dollar per million tokens, three orders of magnitude below any
rate we expect to meet.

Cached input tokens are priced by their own rate, which is the entire reason ``TokenUsage``
counts them apart. One honest limitation follows the one in that module. Anthropic bills
cache *writes* at a premium and cache *reads* at a discount, while ``TokenUsage`` has a
single ``cached_input_tokens`` field documented as what was cached. ``per_cached_input_token``
is therefore the read rate, and an adapter facing a cache-write charge folds those tokens
into ``input_tokens``, where they are billed at the ordinary rate — close to right, and
wrong in the safe direction. Splitting the field is a change to make when an adapter can
demonstrate it matters, not before.

Selecting between price lists by date is not here. A list carries ``effective_from`` so
that a stored cost can be traced back to a set of prices that was current when it was
incurred, but one call must resolve to exactly one list — otherwise ``price_version`` no
longer identifies what was charged. A resolver over several lists belongs to the increment
that first has more than one.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Final, Self

from pydantic import AfterValidator, Field, model_validator

from researchmind.core.base import DomainModel
from researchmind.core.clock import UtcDatetime
from researchmind.core.cost import CallKind, Cost
from researchmind.core.ids import RunId, SubQuestionId
from researchmind.core.money import Money
from researchmind.core.tokens import TokenUsage
from researchmind.providers.completion import MAX_MODEL_NAME_LENGTH, CompletionResult
from researchmind.providers.errors import UnknownModelPriceError

TOKENS_PER_MTOK: Final = 1_000_000
"""The unit vendors quote prices in."""

MAX_PROVIDER_NAME_LENGTH: Final = 64
"""Longest provider name accepted, matching the column ``Cost`` is written to."""

MAX_PRICE_VERSION_LENGTH: Final = 64
"""Longest price list version accepted, matching the column ``Cost`` is written to."""


def _require_non_negative(price: Money) -> Money:
    """Reject a price below zero, whatever Money itself permits.

    Zero is legitimate — a self-hosted vLLM endpoint bills nothing per token — so the bound
    is at zero rather than above it. Negative is not: a rate that pays us to generate would
    make every budget check meaningless in a way no downstream consumer checks for.
    """
    if price.nanodollars < 0:
        msg = "a price cannot be negative"
        raise ValueError(msg)
    return price


PerTokenPrice = Annotated[Money, AfterValidator(_require_non_negative)]
"""What one token costs, as an exact amount rather than a rate to be divided later."""


class ModelPrice(DomainModel):
    """What one model from one provider charges, per token, by kind of token."""

    provider: str = Field(min_length=1, max_length=MAX_PROVIDER_NAME_LENGTH)
    model: str = Field(min_length=1, max_length=MAX_MODEL_NAME_LENGTH)
    per_input_token: PerTokenPrice
    per_cached_input_token: PerTokenPrice
    per_output_token: PerTokenPrice

    @classmethod
    def from_dollars_per_mtok(
        cls,
        *,
        provider: str,
        model: str,
        input_dollars_per_mtok: Decimal | int | str,
        cached_input_dollars_per_mtok: Decimal | int | str,
        output_dollars_per_mtok: Decimal | int | str,
    ) -> ModelPrice:
        """Build a price from the per-million-token dollars a vendor publishes.

        ``float`` is excluded from the signature for the reason ``Money.from_dollars``
        excludes it: a float would reintroduce the representation error the whole money
        type exists to avoid, silently and at the point where the number still looks right.

        A quoted rate finer than a thousandth of a dollar per million tokens is rounded
        half to even, once, here. No vendor currently quotes one.
        """
        return cls(
            provider=provider,
            model=model,
            per_input_token=_per_token(input_dollars_per_mtok),
            per_cached_input_token=_per_token(cached_input_dollars_per_mtok),
            per_output_token=_per_token(output_dollars_per_mtok),
        )

    def charge_for(self, usage: TokenUsage) -> Money:
        """Return what this usage costs at these prices, exactly."""
        return (
            self.per_input_token * usage.input_tokens
            + self.per_cached_input_token * usage.cached_input_tokens
            + self.per_output_token * usage.output_tokens
        )

    def upper_bound(self, *, input_tokens: int, max_tokens: int) -> Money:
        """Return the most a call with this shape can cost (ADR-0004).

        This is the reservation the budget enforcer of phase 4 takes before a call it has
        not yet paid for: the counted input at the input rate, plus the whole of
        ``max_tokens`` at the output rate, on the assumption that generation runs to the
        limit. The bound is conservative in the one way that matters — it prices every
        input token at the full rate, so a call that turns out to hit the prompt cache
        costs less than was held for it, never more.

        The arithmetic is exact rather than rounded up, because per-token prices times
        whole token counts leave nothing to round.
        """
        if input_tokens < 0 or max_tokens < 0:
            msg = "token counts in a reservation cannot be negative"
            raise ValueError(msg)
        return self.per_input_token * input_tokens + self.per_output_token * max_tokens


class PriceList(DomainModel):
    """A set of prices, named by a version and current from a stated instant.

    Lookup is a scan rather than a dictionary built alongside the tuple. A price list holds
    a handful of models, and an index would be a second representation of the same facts
    with nothing keeping it in step.
    """

    version: str = Field(min_length=1, max_length=MAX_PRICE_VERSION_LENGTH)
    effective_from: UtcDatetime
    prices: tuple[ModelPrice, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_a_model_priced_twice(self) -> Self:
        """Check that each provider and model pair appears at most once.

        Two prices for one model make the cost of a call depend on which was found first,
        which is a difference no reader of the resulting row could ever see.
        """
        seen: set[tuple[str, str]] = set()
        for price in self.prices:
            key = (price.provider, price.model)
            if key in seen:
                msg = f"{price.model} from {price.provider} is priced twice in {self.version}"
                raise ValueError(msg)
            seen.add(key)
        return self

    def price_for(self, provider: str, model: str) -> ModelPrice:
        """Return the price of a model.

        Raises:
            UnknownModelPriceError: if this list does not price that model. A missing price
                is a configuration defect of ours, and defaulting it to zero would report a
                run as free instead of reporting that we cannot say what it cost.
        """
        for price in self.prices:
            if price.provider == provider and price.model == model:
                return price
        raise UnknownModelPriceError(provider=provider, model=model, price_version=self.version)

    def cost_for_completion(
        self,
        result: CompletionResult,
        *,
        provider: str,
        run_id: RunId,
        incurred_at: UtcDatetime,
        sub_question_id: SubQuestionId | None = None,
    ) -> Cost:
        """Price one completion and return the record of what it cost.

        The lookup uses ``result.model`` — the model that actually served the call — and not
        the one the request asked for. Aliases resolve and deployments get pointed
        elsewhere, so pricing the requested name would produce a number attributed to a
        model that did no work.

        ``incurred_at`` is passed in rather than read from a clock here. The domain takes
        its instants from callers so that a replayed run (ADR-0005) reproduces the times it
        originally recorded instead of stamping the moment of the replay.

        Raises:
            UnknownModelPriceError: if this list does not price the model that served the
                call.
        """
        price = self.price_for(provider, result.model)
        return Cost(
            call_id=result.call_id,
            run_id=run_id,
            sub_question_id=sub_question_id,
            kind=CallKind.LLM,
            provider=provider,
            model=result.model,
            tokens=result.usage,
            amount=price.charge_for(result.usage),
            latency=result.latency,
            price_version=self.version,
            incurred_at=incurred_at,
        )


def _per_token(dollars_per_mtok: Decimal | int | str) -> Money:
    """Convert a published per-million-token rate into the price of one token."""
    return Money.from_dollars(Decimal(dollars_per_mtok) / TOKENS_PER_MTOK)
