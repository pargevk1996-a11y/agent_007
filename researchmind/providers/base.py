"""The interface every LLM adapter satisfies, and the guarantee it must declare.

ADR-0003 keeps this to the smallest surface that still lets a provider be a configuration
choice, and insists that the one thing vendors genuinely differ about stays visible.
``structured_output_mode`` is that thing. Anthropic and OpenAI constrain generation through
native tool use or a JSON schema mode; vLLM produces valid schemas with guided decoding and
otherwise merely tries. An interface that hid the difference would make the weakest case
invisible, so each adapter declares which guarantee it actually offers and the retry policy
of a later increment is derived from that declaration.

This is a ``Protocol`` rather than an abstract base class. An adapter conforms by having
the right shape, not by inheriting, which keeps the dependency pointing one way and makes
a test double a plain class. The trade is that conformance is checked by ``mypy --strict``
and not at runtime: the protocol is deliberately not ``runtime_checkable``, because an
``isinstance`` check over a protocol tests for the presence of attributes and would report
a provider with a wrongly typed ``complete`` as conforming.

``complete_structured`` is the second method, and it makes one attempt. Retrying is not
part of it: what to do about a failure is decided by the taxonomy's ``retryable`` and by
``structured_output_mode``, and that decision applies to both methods equally. Building it
into one of them would bury a policy that belongs above both.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from researchmind.core.base import DomainModel
from researchmind.providers.completion import CompletionRequest, CompletionResult
from researchmind.providers.structured import StructuredRequest, StructuredResult


class StructuredOutputMode(Enum):
    """How strongly an adapter can constrain a model to a schema (ADR-0003).

    The order is meaningful: each mode offers less than the one before it. ``PROMPTED`` is
    the honest label for asking nicely and hoping, and an adapter that reports it is
    telling the retry policy to expect failures the others do not see.
    """

    NATIVE_TOOLS = "native_tools"
    JSON_SCHEMA = "json_schema"
    GUIDED_DECODING = "guided_decoding"
    PROMPTED = "prompted"

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``guided_decoding``."""
        return self.value


class LLMProvider(Protocol):
    """A completion provider, whichever vendor is behind it."""

    @property
    def name(self) -> str:
        """Return the provider's name as it is recorded against every cost row."""
        ...

    @property
    def structured_output_mode(self) -> StructuredOutputMode:
        """Return the strongest schema guarantee this adapter can offer."""
        ...

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one completion.

        Raises:
            ProviderError: for any failure attributable to the provider, classified by
                the taxonomy in ``researchmind.providers.errors``.
        """
        ...

    async def complete_structured[T: DomainModel](
        self, request: StructuredRequest[T]
    ) -> StructuredResult[T]:
        """Run one completion that must answer with an instance of the request's schema.

        Output that does not validate is reported, not raised: the result carries no value
        and the usage that has to be paid for either way. How hard an adapter tries to
        prevent that is what ``structured_output_mode`` declares.

        Raises:
            ProviderError: for any failure attributable to the provider, classified by
                the taxonomy in ``researchmind.providers.errors``. A model that answered
                badly is not one of them.
        """
        ...
