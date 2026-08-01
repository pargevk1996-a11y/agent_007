"""The Anthropic adapter: our request shape in, our result shape out, their SDK between.

This is the first module that imports a vendor SDK, and the ``import-linter`` contract that
confines vendor SDKs to ``providers`` finally has something to confine. The module is named
``anthropic_adapter`` rather than ``anthropic`` for the reason ``clock`` is not named
``time``: absolute imports would in fact resolve the vendor package correctly, but a module
that shadows the name of the dependency it imports is a trap laid for the next reader.

Three vendor facts are load-bearing here, and each is enforced rather than commented.

**Temperature is not a parameter of the current models.** ``claude-opus-5``,
``claude-sonnet-5``, ``claude-opus-4-8`` and the rest of that family removed
``temperature``, ``top_p`` and ``top_k``; sending one is a 400, not a warning. Our
``CompletionRequest`` still carries a temperature because OpenAI and vLLM still take one,
so the adapter omits it when it is the default and refuses the call when it is not. The
refusal is deliberate: quietly dropping a parameter the caller asked for would make a
request for varied sampling look like it succeeded.

**Cache writes are billed as ordinary input.** The provider reports cached reads and cache
creation separately. ``TokenUsage`` counts cached tokens apart because they are *cheaper*,
and cache creation is not — it is dearer than plain input. Folding creation into
``input_tokens`` prices it at the ordinary rate: close to right, and wrong in the direction
that under-reports a premium rather than pricing a premium as a discount. ``pricing`` states
the same limitation from the other side.

**Thinking is on by default on the current models, and it spends output tokens.** No
``thinking`` parameter is sent, so the model decides. That leaves the accounting sound —
thinking tokens are output tokens, so ADR-0004's reservation, which prices the whole of
``max_tokens`` at the output rate, still bounds the call — but it means a ``max_tokens``
sized tightly around the expected answer will truncate. That surfaces as
``StopReason.MAX_TOKENS``, which is exactly the signal that type exists to carry.

The client is injected rather than constructed here, because ``config`` is the only reader
of the environment. It is expected to have been built with ``max_retries=0``. The SDK
retries rate limits and server errors by default, and a retry we did not decide to make is
a retry the policy of the next increment cannot see, cannot budget for and cannot record —
ADR-0005 wants every call replayable, and a call that silently happened three times is not.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from typing import Final

import anthropic
from anthropic import AsyncAnthropic, Omit, omit
from anthropic.lib._parse._transform import transform_schema
from anthropic.types import JSONOutputFormatParam, Message, MessageParam, OutputConfigParam
from pydantic import ValidationError

from researchmind.core.base import DomainModel
from researchmind.core.tokens import TokenUsage
from researchmind.providers.base import StructuredOutputMode
from researchmind.providers.completion import CompletionRequest, CompletionResult, StopReason
from researchmind.providers.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from researchmind.providers.structured import StructuredRequest, StructuredResult

PROVIDER_NAME: Final = "anthropic"
"""The name recorded against every cost row this adapter produces."""

MODELS_WITHOUT_TEMPERATURE: Final = frozenset(
    {
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
    }
)
"""Models that reject a sampling temperature outright rather than ignoring it."""

_STOP_REASONS: Final = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.REFUSAL,
    "model_context_window_exceeded": StopReason.CONTEXT_WINDOW_EXCEEDED,
}
"""The vendor's stop reasons that this contract has a meaning for.

``tool_use`` and ``pause_turn`` are absent on purpose. We send no tools, so neither can
arise from a request we built; if one arrives, the response is not the one we asked for and
reporting it as a completed turn would hand the caller an answer the model never finished.
"""


class AnthropicProvider:
    """An :class:`~researchmind.providers.base.LLMProvider` backed by Anthropic.

    Conformance is by shape, not by inheritance: ``mypy --strict`` checks this class against
    the protocol at every call site that expects one.
    """

    def __init__(self, client: AsyncAnthropic) -> None:
        """Wrap a configured client, which is expected to do no retrying of its own."""
        self._client = client

    @property
    def name(self) -> str:
        """Return the provider's name as it is recorded against every cost row."""
        return PROVIDER_NAME

    @property
    def structured_output_mode(self) -> StructuredOutputMode:
        """Return ``NATIVE_TOOLS``: generation is constrained by the vendor, not by us."""
        return StructuredOutputMode.NATIVE_TOOLS

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one completion.

        Raises:
            ProviderError: for any failure attributable to the provider, classified by the
                taxonomy in ``researchmind.providers.errors``.
        """
        message, latency = await self._send(request)
        return CompletionResult(
            call_id=request.call_id,
            model=message.model,
            text=_text_of(message),
            stop_reason=_stop_reason_of(message),
            usage=_usage_of(message),
            latency=latency,
        )

    async def complete_structured[T: DomainModel](
        self, request: StructuredRequest[T]
    ) -> StructuredResult[T]:
        """Run one completion constrained to the request's schema.

        The vendor's own ``messages.parse`` helper is not used, and the reason is not
        stylistic: it validates inside the response parser, so output that does not
        validate raises out of the HTTP call and takes the whole message with it — usage
        included. The call was made and billed; a caller left holding an exception has
        nothing to commit against the budget. Validating here keeps the accounting whether
        or not the model answered well.

        Raises:
            ProviderError: for any failure attributable to the provider. Output that does
                not validate is not one: it comes back as a result with no value.
        """
        message, latency = await self._send(request, schema=request.output_schema)
        text = _text_of(message)
        stop_reason = _stop_reason_of(message)
        return StructuredResult[T](
            call_id=request.call_id,
            model=message.model,
            text=text,
            stop_reason=stop_reason,
            usage=_usage_of(message),
            latency=latency,
            value=_value_of(text, request.output_schema, stop_reason),
        )

    async def _send(
        self, request: CompletionRequest, *, schema: type[DomainModel] | None = None
    ) -> tuple[Message, timedelta]:
        """Make the call, and translate every way it can fail.

        Shared by both methods so that a failure means the same thing whichever one asked.
        """
        temperature = self._temperature_for(request)
        started = time.monotonic()
        try:
            payload: object = await self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                messages=[
                    MessageParam(role=turn.role.value, content=turn.content)
                    for turn in request.messages
                ],
                system=request.system if request.system is not None else omit,
                temperature=temperature,
                output_config=_output_config(schema),
            )
        except anthropic.APITimeoutError as exc:
            # Before APIConnectionError, which it subclasses. The distinction is the whole
            # point: a timeout may still be running and may still be billed.
            raise ProviderTimeoutError(str(exc), provider=PROVIDER_NAME) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise ProviderAuthError(
                exc.message, provider=PROVIDER_NAME, status=exc.status_code
            ) from exc
        except (
            anthropic.BadRequestError,
            anthropic.NotFoundError,
            anthropic.UnprocessableEntityError,
            anthropic.RequestTooLargeError,
        ) as exc:
            raise ProviderRequestError(
                exc.message, provider=PROVIDER_NAME, status=exc.status_code
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(
                exc.message,
                provider=PROVIDER_NAME,
                status=exc.status_code,
                retry_after=_retry_after(exc),
            ) from exc
        except (anthropic.OverloadedError, anthropic.InternalServerError) as exc:
            raise ProviderUnavailableError(
                exc.message, provider=PROVIDER_NAME, status=exc.status_code
            ) from exc
        except anthropic.APIConnectionError as exc:
            # No response arrived and none is coming. The same request may well succeed.
            raise ProviderUnavailableError(str(exc), provider=PROVIDER_NAME) from exc
        except anthropic.APIStatusError as exc:
            # A status we have no meaning for. Not classified as retryable, because
            # guessing wrong in that direction repeats a call that costs money.
            raise ProviderResponseError(
                exc.message, provider=PROVIDER_NAME, status=exc.status_code
            ) from exc
        except anthropic.APIResponseValidationError as exc:
            raise ProviderResponseError(str(exc), provider=PROVIDER_NAME) from exc
        except json.JSONDecodeError as exc:
            # Not a vendor exception at all: a body that claims to be JSON and is not
            # surfaces as a ValueError from the parser, which would otherwise escape the
            # error hierarchy entirely on its way up.
            msg = "the provider sent a body that claimed to be JSON and was not"
            raise ProviderResponseError(msg, provider=PROVIDER_NAME) from exc
        latency = timedelta(seconds=time.monotonic() - started)
        return _require_a_message(payload), latency

    def _temperature_for(self, request: CompletionRequest) -> float | Omit:
        """Decide what to send as the temperature, or refuse to send the request.

        Raises:
            ProviderRequestError: if a temperature was chosen for a model that has no such
                parameter. Ours to fix, and not retryable: the same request fails the same
                way until the caller stops asking for it.
        """
        if request.model not in MODELS_WITHOUT_TEMPERATURE:
            return request.temperature
        if request.temperature != 0.0:
            msg = (
                f"model {request.model!r} does not accept a temperature; "
                f"{request.temperature} was requested"
            )
            raise ProviderRequestError(msg, provider=PROVIDER_NAME)
        return omit


def _output_config(schema: type[DomainModel] | None) -> OutputConfigParam | Omit:
    """Turn a domain model into the schema the vendor will constrain generation to.

    ``transform_schema`` is the SDK's own, and is reached through a private module. That is
    normally a smell, and it is the lesser one here: it is the transform the vendor's
    ``parse`` helper applies, so the schema we send is the schema the SDK would have sent,
    and reimplementing it would mean maintaining a second opinion about a dialect we do not
    own. The version is pinned exactly and upgrades are read as lock diffs, so a change to
    it arrives as a review rather than as a surprise — and a test pins what it produces.

    Two of its behaviours are worth knowing at the call site. It adds
    ``additionalProperties: false``, which our models already imply. And it moves the
    constraints the dialect does not support — string lengths, numeric bounds — into the
    schema's prose description, where they advise the model rather than binding it. Those
    constraints are still ours to enforce, at parse time, which is why an extraction can
    come back empty from a call that ended perfectly normally.
    """
    if schema is None:
        return omit
    return OutputConfigParam(
        format=JSONOutputFormatParam(type="json_schema", schema=transform_schema(schema))
    )


def _value_of[T: DomainModel](text: str, schema: type[T], stop_reason: StopReason) -> T | None:
    """Parse the answer, or report that there was not one to parse.

    A generation that did not run to its own end has nothing complete to offer, and trying
    would at best produce an object built from a truncated document. Anything else that
    fails to validate is reported as an absent value rather than raised, because the call
    has already been paid for.
    """
    if stop_reason is not StopReason.END_TURN:
        return None
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        return None


def _require_a_message(payload: object) -> Message:
    """Check that what came back is a message, and not merely typed as one.

    The SDK parses leniently, and the type annotations do not say so. A response body that
    is not JSON at all is returned as the raw string; a body of well-formed JSON with the
    wrong shape is returned as a ``Message`` whose every field is ``None``. Neither raises,
    so without this the first symptom is an ``AttributeError`` from inside the mapping,
    reported as a bug in code that did nothing wrong.

    Revalidating the parsed object strictly turns both into the typed response failure the
    taxonomy already has a place for. The round trip is faithful: a well-formed message
    revalidates to an equal object, content blocks included.

    Raises:
        ProviderResponseError: if the payload is not a well-formed message.
    """
    if not isinstance(payload, Message):
        msg = f"the provider returned {type(payload).__name__}, which is not a message"
        raise ProviderResponseError(msg, provider=PROVIDER_NAME)
    try:
        return Message.model_validate(payload.model_dump())
    except ValidationError as exc:
        msg = "the provider returned a message with fields missing or of the wrong type"
        raise ProviderResponseError(msg, provider=PROVIDER_NAME) from exc


def _text_of(message: Message) -> str:
    """Join the text the model produced, ignoring blocks that are not text.

    Thinking blocks arrive here when the model reasons, and they are not the answer. The
    result may legitimately be empty; ``CompletionResult`` permits that.
    """
    return "".join(block.text for block in message.content if block.type == "text")


def _stop_reason_of(message: Message) -> StopReason:
    """Translate the vendor's stop reason into ours.

    Raises:
        ProviderResponseError: for a stop reason this contract has no meaning for,
            including its absence.
    """
    reason = _STOP_REASONS.get(message.stop_reason or "")
    if reason is None:
        msg = f"unhandled stop reason {message.stop_reason!r} from a request that sent no tools"
        raise ProviderResponseError(msg, provider=PROVIDER_NAME)
    return reason


def _usage_of(message: Message) -> TokenUsage:
    """Translate the vendor's usage report, pricing cache writes as ordinary input."""
    usage = message.usage
    return TokenUsage(
        input_tokens=usage.input_tokens + (usage.cache_creation_input_tokens or 0),
        cached_input_tokens=usage.cache_read_input_tokens or 0,
        output_tokens=usage.output_tokens,
    )


def _retry_after(error: anthropic.RateLimitError) -> timedelta | None:
    """Read the wait the provider asked for, if it asked for one in a form we understand.

    An unparseable or negative header is treated as absent rather than as zero: "it did not
    say" is the honest reading, and zero would tell the retry policy to go straight back.
    """
    header = error.response.headers.get("retry-after")
    if header is None:
        return None
    try:
        seconds = float(header)
    except ValueError:
        # The header also permits an HTTP date. Parsing one needs a clock to subtract it
        # from, which this function does not have and should not acquire.
        return None
    if seconds < 0:
        return None
    return timedelta(seconds=seconds)
