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
from pydantic import Field

from researchmind.core.base import DomainModel
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
from researchmind.providers.structured import StructuredRequest

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


class Finding(DomainModel):
    """An extraction target with a bound the vendor's schema will not enforce."""

    headline: str = Field(min_length=1, max_length=40)
    confident: bool


def _structured_request(*, model: str = "claude-opus-5") -> StructuredRequest[Finding]:
    return StructuredRequest[Finding](
        call_id=new_call_id(),
        model=model,
        messages=(Message(role=Role.USER, content="Summarise the finding."),),
        max_tokens=1024,
        output_schema=Finding,
    )


def _json_body(text: str, *, stop_reason: str = "end_turn") -> dict[str, object]:
    return _body(content=[{"type": "text", "text": text}], stop_reason=stop_reason)


@respx.mock
async def test_a_schema_is_sent_as_the_output_config() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_json_body('{"headline": "MiCA", "confident": true}'))
    )

    await _provider().complete_structured(_structured_request())

    sent = json.loads(route.calls.last.request.content)["output_config"]["format"]
    assert sent["type"] == "json_schema"
    assert sent["schema"]["required"] == ["headline", "confident"]


@respx.mock
async def test_a_schema_forbids_the_fields_it_did_not_ask_for() -> None:
    # The vendor requires additionalProperties: false on every object, which our domain
    # models already imply by being closed.
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_json_body('{"headline": "MiCA", "confident": true}'))
    )

    await _provider().complete_structured(_structured_request())

    schema = json.loads(route.calls.last.request.content)["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False


@respx.mock
async def test_a_length_bound_reaches_the_model_as_advice_not_as_a_constraint() -> None:
    # The dialect has no maxLength, so the SDK's transform moves it into the description.
    # This is why an extraction can come back empty from a turn that ended normally, and
    # pins the behaviour so a lock bump that changes it fails here rather than in a run.
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_json_body('{"headline": "MiCA", "confident": true}'))
    )

    await _provider().complete_structured(_structured_request())

    schema = json.loads(route.calls.last.request.content)["output_config"]["format"]["schema"]
    headline = schema["properties"]["headline"]
    assert "maxLength" not in headline
    assert "maxLength: 40" in headline["description"]


@respx.mock
async def test_a_well_formed_answer_comes_back_as_the_type_that_was_asked_for() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, json=_json_body('{"headline": "MiCA applies", "confident": true}')
        )
    )

    result = await _provider().complete_structured(_structured_request())

    assert result.extracted
    assert result.value == Finding(headline="MiCA applies", confident=True)
    assert result.stop_reason is StopReason.END_TURN
    assert result.usage.input_tokens == 1000


@respx.mock
async def test_an_answer_that_breaks_a_bound_the_schema_could_not_carry_costs_us_anyway() -> None:
    # 60 characters where the type permits 40. The generation was never constrained, so
    # this is a completed, billed call that produced nothing usable.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, json=_json_body(json.dumps({"headline": "M" * 60, "confident": True}))
        )
    )

    result = await _provider().complete_structured(_structured_request())

    assert not result.extracted
    assert result.stop_reason is StopReason.END_TURN
    assert result.usage.output_tokens == 50
    assert "MMM" in result.text


@respx.mock
async def test_an_answer_that_is_not_the_shape_we_asked_for_is_reported_not_raised() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(200, json=_json_body('{"headline": "MiCA"}'))
    )

    result = await _provider().complete_structured(_structured_request())

    assert not result.extracted
    assert result.usage.total == 1050


@respx.mock
async def test_an_answer_that_is_not_json_at_all_is_reported_not_raised() -> None:
    respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_json_body("I cannot.")))
    result = await _provider().complete_structured(_structured_request())
    assert not result.extracted
    assert result.text == "I cannot."


@pytest.mark.parametrize("reported", ["refusal", "max_tokens", "model_context_window_exceeded"])
@respx.mock
async def test_a_generation_that_did_not_finish_yields_no_value_and_still_reports_usage(
    reported: str,
) -> None:
    # A truncated document is not parsed even when the prefix happens to be valid JSON.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            json=_json_body(
                '{"headline": "MiCA applies", "confident": true}', stop_reason=reported
            ),
        )
    )

    result = await _provider().complete_structured(_structured_request())

    assert not result.extracted
    assert result.usage.total == 1050


@respx.mock
async def test_a_structured_call_fails_the_way_an_unstructured_one_does() -> None:
    # Both methods send through the same translation, so a rate limit means the same thing
    # whichever one asked.
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            429, json=_error("rate_limit_error", "slow down"), headers={"retry-after": "3"}
        )
    )

    with pytest.raises(ProviderRateLimitError) as caught:
        await _provider().complete_structured(_structured_request())

    assert caught.value.retry_after == timedelta(seconds=3)


@respx.mock
async def test_a_structured_call_refuses_a_temperature_the_model_will_not_take() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))

    with pytest.raises(ProviderRequestError, match="does not accept a temperature"):
        await _provider().complete_structured(
            StructuredRequest[Finding](
                call_id=new_call_id(),
                model="claude-opus-5",
                messages=(Message(role=Role.USER, content="Summarise it."),),
                max_tokens=1024,
                temperature=0.9,
                output_schema=Finding,
            )
        )

    assert not route.called


@respx.mock
async def test_an_unstructured_call_sends_no_output_config() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _provider().complete(_request())

    assert "output_config" not in json.loads(route.calls.last.request.content)
