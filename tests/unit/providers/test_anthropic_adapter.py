"""The Anthropic adapter, driven over a mocked transport rather than a hand-built double.

The vendor SDK builds the request and parses the response exactly as it would in
production, and raises its own exception classes; only the socket is fake. A stub client
would test our idea of the SDK, which is the thing most likely to be wrong.

Every client here is built with ``max_retries=0``, which is what the adapter documents it
expects. It is also what keeps these tests honest: with retries on, the SDK would quietly
replay a 429 three times and the assertions would still pass.
"""

import json
from datetime import timedelta

import anthropic
import httpx
import pytest
import respx
from anthropic import AsyncAnthropic

from researchmind.core.ids import new_call_id
from researchmind.providers.anthropic_adapter import (
    MODELS_WITHOUT_TEMPERATURE,
    PROVIDER_NAME,
    AnthropicProvider,
)
from researchmind.providers.base import StructuredOutputMode
from researchmind.providers.completion import CompletionRequest, Message, Role, StopReason
from researchmind.providers.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def _provider() -> AnthropicProvider:
    return AnthropicProvider(AsyncAnthropic(api_key="test-key", max_retries=0))


def _request(
    *,
    model: str = "claude-opus-5",
    system: str | None = None,
    temperature: float = 0.0,
    messages: tuple[Message, ...] | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        call_id=new_call_id(),
        model=model,
        messages=messages
        if messages is not None
        else (Message(role=Role.USER, content="Compare stablecoin rules."),),
        system=system,
        max_tokens=1024,
        temperature=temperature,
    )


def _body(
    *,
    model: str = "claude-opus-5",
    content: list[dict[str, object]] | None = None,
    stop_reason: str | None = "end_turn",
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content if content is not None else [{"type": "text", "text": "Here it is."}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage if usage is not None else {"input_tokens": 1000, "output_tokens": 50},
    }


def _error(kind: str, message: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def test_the_adapter_declares_who_it_is_and_what_it_guarantees() -> None:
    provider = _provider()
    assert provider.name == PROVIDER_NAME
    assert provider.structured_output_mode is StructuredOutputMode.NATIVE_TOOLS


@respx.mock
async def test_a_completion_comes_back_as_our_result() -> None:
    respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))
    request = _request()

    result = await _provider().complete(request)

    assert result.call_id == request.call_id
    assert result.model == "claude-opus-5"
    assert result.text == "Here it is."
    assert result.stop_reason is StopReason.END_TURN
    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 50
    assert result.latency >= timedelta(0)


@respx.mock
async def test_the_model_that_served_the_call_is_the_one_reported() -> None:
    # An alias resolved. Pricing and evaluation both key off the served name.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_body(model="claude-opus-5-20260101"))
    )
    result = await _provider().complete(_request())
    assert result.model == "claude-opus-5-20260101"


@respx.mock
async def test_the_system_prompt_is_sent_as_a_parameter_not_a_message() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _provider().complete(_request(system="You are a careful researcher."))

    sent = json.loads(route.calls.last.request.content)
    assert sent["system"] == "You are a careful researcher."
    assert [turn["role"] for turn in sent["messages"]] == ["user"]


@respx.mock
async def test_an_absent_system_prompt_is_omitted_rather_than_sent_as_null() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _provider().complete(_request())

    assert "system" not in json.loads(route.calls.last.request.content)


@respx.mock
async def test_an_exchange_is_sent_in_order() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _provider().complete(
        _request(
            messages=(
                Message(role=Role.USER, content="And the EU?"),
                Message(role=Role.ASSISTANT, content="MiCA applies."),
                Message(role=Role.USER, content="Since when?"),
            )
        )
    )

    sent = json.loads(route.calls.last.request.content)
    assert [(turn["role"], turn["content"]) for turn in sent["messages"]] == [
        ("user", "And the EU?"),
        ("assistant", "MiCA applies."),
        ("user", "Since when?"),
    ]


@respx.mock
async def test_no_temperature_is_sent_to_a_model_that_has_no_such_parameter() -> None:
    # Sending one is a 400 on these models, not a warning.
    route = respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _provider().complete(_request(model="claude-opus-5"))

    assert "temperature" not in json.loads(route.calls.last.request.content)


@respx.mock
async def test_a_requested_temperature_is_refused_rather_than_dropped() -> None:
    # Silently discarding it would make a request for varied sampling look like it worked.
    route = respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))

    with pytest.raises(ProviderRequestError, match="does not accept a temperature"):
        await _provider().complete(_request(model="claude-opus-5", temperature=0.7))

    assert not route.called


@respx.mock
async def test_a_temperature_is_sent_to_a_model_that_takes_one() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_body(model="claude-haiku-4-5"))
    )

    await _provider().complete(_request(model="claude-haiku-4-5", temperature=0.7))

    assert json.loads(route.calls.last.request.content)["temperature"] == 0.7


def test_the_models_without_a_temperature_are_named_not_guessed() -> None:
    # A prefix match would classify a future model by its name rather than its contract.
    assert "claude-opus-5" in MODELS_WITHOUT_TEMPERATURE
    assert "claude-haiku-4-5" not in MODELS_WITHOUT_TEMPERATURE


@respx.mock
async def test_cached_reads_are_counted_apart_from_ordinary_input() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            json=_body(
                usage={
                    "input_tokens": 100,
                    "cache_read_input_tokens": 9000,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 50,
                }
            ),
        )
    )

    usage = (await _provider().complete(_request())).usage

    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 9000


@respx.mock
async def test_cache_writes_are_counted_as_ordinary_input() -> None:
    # They are dearer than plain input, not cheaper. Counting them as cached would price a
    # premium as a discount; counting them as input under-reports the premium instead.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            json=_body(
                usage={
                    "input_tokens": 100,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 9000,
                    "output_tokens": 50,
                }
            ),
        )
    )

    usage = (await _provider().complete(_request())).usage

    assert usage.input_tokens == 9100
    assert usage.cached_input_tokens == 0


@respx.mock
async def test_unreported_cache_counts_are_zero_rather_than_an_error() -> None:
    respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))
    usage = (await _provider().complete(_request())).usage
    assert usage.cached_input_tokens == 0


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("stop_sequence", StopReason.STOP_SEQUENCE),
        ("refusal", StopReason.REFUSAL),
        ("model_context_window_exceeded", StopReason.CONTEXT_WINDOW_EXCEEDED),
    ],
)
@respx.mock
async def test_each_stop_reason_we_understand_survives_the_crossing(
    reported: str, expected: StopReason
) -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_body(stop_reason=reported))
    )
    result = await _provider().complete(_request())
    assert result.stop_reason is expected


@respx.mock
async def test_a_refusal_still_reports_what_it_cost() -> None:
    # The call succeeded and was billed. Raising here would lose the usage the budget has
    # to commit either way.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_body(content=[], stop_reason="refusal"))
    )

    result = await _provider().complete(_request())

    assert result.stop_reason is StopReason.REFUSAL
    assert result.text == ""
    assert result.usage.output_tokens == 50


@pytest.mark.parametrize("reported", ["tool_use", "pause_turn", None])
@respx.mock
async def test_a_stop_reason_we_cannot_mean_is_refused(reported: str | None) -> None:
    # We sent no tools, so neither can follow from the request we built.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_body(stop_reason=reported))
    )
    with pytest.raises(ProviderResponseError, match="unhandled stop reason"):
        await _provider().complete(_request())


@respx.mock
async def test_blocks_that_are_not_text_are_not_part_of_the_answer() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            json=_body(
                content=[
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": "The answer."},
                ]
            ),
        )
    )
    assert (await _provider().complete(_request())).text == "The answer."


@respx.mock
async def test_an_empty_completion_is_a_result_not_a_failure() -> None:
    respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body(content=[])))
    assert (await _provider().complete(_request())).text == ""


@pytest.mark.parametrize(
    ("status", "kind", "expected"),
    [
        (401, "authentication_error", ProviderAuthError),
        (403, "permission_error", ProviderAuthError),
        (400, "invalid_request_error", ProviderRequestError),
        (404, "not_found_error", ProviderRequestError),
        (413, "request_too_large", ProviderRequestError),
        (422, "invalid_request_error", ProviderRequestError),
        (500, "api_error", ProviderUnavailableError),
        (529, "overloaded_error", ProviderUnavailableError),
        (409, "conflict_error", ProviderResponseError),
    ],
)
@respx.mock
async def test_each_status_lands_in_the_part_of_the_taxonomy_that_says_what_to_do(
    status: int, kind: str, expected: type[Exception]
) -> None:
    respx.post(MESSAGES_URL).mock(return_value=httpx.Response(status, json=_error(kind, "no")))

    with pytest.raises(expected) as caught:
        await _provider().complete(_request())

    assert caught.value.provider == PROVIDER_NAME  # type: ignore[attr-defined]  # narrowed by the taxonomy
    assert caught.value.status == status  # type: ignore[attr-defined]  # as above


@respx.mock
async def test_rate_limiting_carries_the_wait_the_provider_asked_for() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            429, json=_error("rate_limit_error", "slow down"), headers={"retry-after": "12"}
        )
    )

    with pytest.raises(ProviderRateLimitError) as caught:
        await _provider().complete(_request())

    assert caught.value.retry_after == timedelta(seconds=12)
    assert caught.value.retryable is True


@respx.mock
async def test_an_unstated_wait_is_absent_rather_than_zero() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(429, json=_error("rate_limit_error", "slow down"))
    )
    with pytest.raises(ProviderRateLimitError) as caught:
        await _provider().complete(_request())
    assert caught.value.retry_after is None


@pytest.mark.parametrize("header", ["Sat, 01 Aug 2026 12:00:00 GMT", "-5", "soon"])
@respx.mock
async def test_a_wait_we_cannot_read_is_absent_rather_than_zero(header: str) -> None:
    # Zero would tell the retry policy to go straight back, which is the one thing the
    # provider was asking us not to do.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            429, json=_error("rate_limit_error", "slow down"), headers={"retry-after": header}
        )
    )
    with pytest.raises(ProviderRateLimitError) as caught:
        await _provider().complete(_request())
    assert caught.value.retry_after is None


@respx.mock
async def test_a_timeout_is_not_reported_as_a_connection_failure() -> None:
    # The SDK's timeout subclasses its connection error, and the two mean different things:
    # a timed-out call may still be running and may still be billed.
    respx.post(MESSAGES_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    with pytest.raises(ProviderTimeoutError) as caught:
        await _provider().complete(_request())

    assert caught.value.status is None
    assert caught.value.retryable is True


@respx.mock
async def test_a_connection_failure_is_treated_as_the_provider_being_unavailable() -> None:
    respx.post(MESSAGES_URL).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(ProviderUnavailableError):
        await _provider().complete(_request())


@respx.mock
async def test_a_body_that_is_not_json_is_a_response_failure() -> None:
    # The SDK hands back the raw string here rather than raising. Without a check of our
    # own, the first symptom is an AttributeError from inside the mapping.
    respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, content=b"not json"))

    with pytest.raises(ProviderResponseError, match="not a message"):
        await _provider().complete(_request())


@respx.mock
async def test_a_body_that_claims_to_be_json_and_is_not_is_a_response_failure() -> None:
    # This one surfaces as a ValueError from the parser, not as a vendor exception, and
    # would otherwise escape the error hierarchy on its way up.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, content=b"not json", headers={"content-type": "application/json"}
        )
    )

    with pytest.raises(ProviderResponseError, match="claimed to be JSON"):
        await _provider().complete(_request())


@respx.mock
async def test_json_of_the_wrong_shape_is_a_response_failure() -> None:
    # The SDK parses this into a Message with every field set to None, and the type
    # annotations give no hint that it can.
    respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json={"hello": "world"}))

    with pytest.raises(ProviderResponseError, match="fields missing or of the wrong type"):
        await _provider().complete(_request())


@respx.mock
async def test_the_adapter_leaves_no_vendor_exception_uncaught() -> None:
    # `except ResearchmindError` has to be enough at the call site, which is only true if
    # nothing from the SDK escapes this module.
    respx.post(MESSAGES_URL).mock(side_effect=httpx.ConnectError("no route"))

    try:
        await _provider().complete(_request())
    except anthropic.AnthropicError as exc:  # pragma: no cover - the failure we are excluding
        pytest.fail(f"a vendor exception escaped the adapter: {exc!r}")
    except ProviderUnavailableError:
        pass
