"""What is sent to a model and what comes back, in one shape for every vendor.

ADR-0003 keeps this interface narrow so that swapping a provider is a configuration
change. Narrow means it carries what every adapter genuinely needs and nothing that only
one of them understands; vendor-specific behaviour lives in the adapter, where it can be
seen.

Two of the fields here are load-bearing beyond their size.

``max_tokens`` is required and has no default. ADR-0004 computes the reservation for a
call as counted input tokens times the input price plus ``max_tokens`` times the output
price, and clamps it to what the remaining budget affords rather than taking it from
configuration unexamined. A default would be exactly the unexamined value the decision
forbids, arriving quietly at every call site that forgot to think.

``call_id`` is minted by the caller and travels in the request. The identifiers module
gives the reason: an object is logged before it is written, and a failed write is the case
worth tracing. If the provider minted it and returned it in the result, a call that failed
before returning would have no identity at all — which is the call you most want to find.

Roles alternate and the exchange starts with the user. That is the strictest of the vendor
rules rather than the most permissive: Anthropic requires it, OpenAI tolerates anything.
Enforcing it here means a request that was built successfully can be executed by any
adapter, instead of failing at whichever one happens to be configured today.

There is no streaming in this interface. A method that returns either a result or an
iterator is two contracts wearing one name; if token streaming is ever needed it arrives
as its own method. The SSE stream of phase 14 carries events, which is a different thing.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from itertools import pairwise
from typing import Final, Self

from pydantic import Field, model_validator

from researchmind.core.base import DomainModel
from researchmind.core.ids import CallId
from researchmind.core.tokens import TokenUsage

MAX_MODEL_NAME_LENGTH: Final = 128
"""Longest model identifier accepted."""

MAX_TEMPERATURE: Final = 2.0
"""The union of the vendors' ranges, not any one vendor's.

Anthropic tops out at 1.0 and OpenAI at 2.0. This type does not know which model a name
refers to, so it cannot narrow further; the adapter can and must.
"""


class Role(Enum):
    """Who is speaking in an exchange.

    There is no system role. The system prompt is a field of the request, because
    Anthropic takes it as a top-level parameter and OpenAI as a message, and mapping
    between the two is the adapter's job. Keeping it out of the sequence also makes a
    system message stranded in the middle of a conversation unrepresentable.
    """

    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``assistant``."""
        return self.value


class StopReason(Enum):
    """Why generation ended.

    ``MAX_TOKENS`` earns its place in the type. With ``max_tokens`` clamped to what the
    budget affords (ADR-0004), truncation stops being an edge case, and a truncated answer
    is a silent quality failure unless the caller is made to see it.
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``max_tokens``."""
        return self.value


class Message(DomainModel):
    """One turn of an exchange.

    ``content`` has no upper bound. Source excerpts go through here, and any ceiling this
    module invented would either obstruct legitimate work or be too high to mean anything.
    What limits the size of a call is the budget, which knows the price of a token.
    """

    role: Role
    content: str = Field(min_length=1)


class CompletionRequest(DomainModel):
    """One completion call, as this system describes it."""

    call_id: CallId
    model: str = Field(min_length=1, max_length=MAX_MODEL_NAME_LENGTH)
    messages: tuple[Message, ...] = Field(min_length=1)
    system: str | None = Field(default=None, min_length=1)
    max_tokens: int = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=MAX_TEMPERATURE)

    @model_validator(mode="after")
    def _require_an_alternating_exchange(self) -> Self:
        """Check that the user speaks first and the turns alternate."""
        if self.messages[0].role is not Role.USER:
            msg = "an exchange must open with a message from the user"
            raise ValueError(msg)
        for earlier, later in pairwise(self.messages):
            if earlier.role is later.role:
                msg = f"two consecutive messages from {earlier.role}; roles must alternate"
                raise ValueError(msg)
        return self


class CompletionResult(DomainModel):
    """What a provider returned, with everything needed to account for it.

    ``model`` is the model that actually served the call, which is not always the one that
    was asked for: aliases resolve, and a deployment can be pointed elsewhere. Recording
    the served name is what makes an evaluation comparable across runs.

    ``text`` may be empty. A model that stops at once has produced an empty completion,
    and requiring content here would only push an adapter into inventing some.

    There is no cost. Cost is computed from ``usage`` against a versioned price list, which
    is the next increment; deriving it here would put the price list in the wrong place.
    """

    call_id: CallId
    model: str = Field(min_length=1, max_length=MAX_MODEL_NAME_LENGTH)
    text: str
    stop_reason: StopReason
    usage: TokenUsage
    latency: timedelta = Field(ge=timedelta(0))
