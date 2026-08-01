"""The shape of a call, and the exchanges the contract refuses to build."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from researchmind.core.ids import new_call_id
from researchmind.core.tokens import TokenUsage
from researchmind.providers.completion import (
    MAX_MODEL_NAME_LENGTH,
    MAX_TEMPERATURE,
    CompletionRequest,
    CompletionResult,
    Message,
    Role,
    StopReason,
)


def _user(content: str = "Compare stablecoin rules across three jurisdictions.") -> Message:
    return Message(role=Role.USER, content=content)


def _assistant(content: str = "Here is the comparison.") -> Message:
    return Message(role=Role.ASSISTANT, content=content)


def _request(
    *,
    messages: tuple[Message, ...] | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> CompletionRequest:
    return CompletionRequest(
        call_id=new_call_id(),
        model="claude-opus-5",
        messages=messages if messages is not None else (_user(),),
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def test_a_single_turn_request_is_accepted() -> None:
    request = _request()
    assert request.messages[0].role is Role.USER
    assert request.temperature == 0.0
    assert request.system is None


def test_a_multi_turn_exchange_alternates() -> None:
    request = _request(messages=(_user(), _assistant(), _user("And Singapore?")))
    assert [message.role for message in request.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]


def test_an_exchange_that_opens_with_the_assistant_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must open with a message from the user"):
        _request(messages=(_assistant(), _user()))


def test_two_consecutive_turns_from_the_same_role_are_rejected() -> None:
    # The strictest of the vendor rules, enforced where the request is built rather than
    # at whichever adapter happens to be configured.
    with pytest.raises(ValidationError, match="roles must alternate"):
        _request(messages=(_user(), _user("And Singapore?")))


def test_an_empty_exchange_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(messages=())


def test_the_system_prompt_is_a_field_and_not_a_role() -> None:
    request = _request(system="You are a research assistant.")
    assert request.system == "You are a research assistant."
    assert all(message.role is not Role.ASSISTANT for message in request.messages[:1])
    assert {role.value for role in Role} == {"user", "assistant"}


@pytest.mark.parametrize("system", ["", "   "])
def test_a_blank_system_prompt_is_rejected(system: str) -> None:
    # Absent is None. A string of spaces is a mistake, not an instruction.
    with pytest.raises(ValidationError):
        _request(system=system)


@pytest.mark.parametrize("content", ["", "   "])
def test_a_blank_message_is_rejected(content: str) -> None:
    with pytest.raises(ValidationError):
        Message(role=Role.USER, content=content)


def test_message_content_has_no_ceiling() -> None:
    # Source excerpts travel through here; what limits a call is the budget, not this type.
    assert len(Message(role=Role.USER, content="x" * 500_000).content) == 500_000


def test_max_tokens_has_no_default() -> None:
    # ADR-0004 clamps max_tokens to what the budget affords rather than taking it from
    # configuration unexamined. A default would be that unexamined value.
    with pytest.raises(ValidationError):
        CompletionRequest.model_validate(
            {
                "call_id": new_call_id(),
                "model": "claude-opus-5",
                "messages": (_user(),),
            }
        )


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_a_call_must_be_allowed_to_produce_something(max_tokens: int) -> None:
    with pytest.raises(ValidationError):
        _request(max_tokens=max_tokens)


def test_temperature_defaults_to_deterministic() -> None:
    assert _request().temperature == 0.0


@pytest.mark.parametrize("temperature", [-0.1, MAX_TEMPERATURE + 0.1])
def test_a_temperature_outside_every_vendor_range_is_rejected(temperature: float) -> None:
    with pytest.raises(ValidationError):
        _request(temperature=temperature)


def test_a_temperature_legal_for_one_vendor_only_still_passes_here() -> None:
    # This type does not know which model a name refers to, so it cannot narrow to
    # Anthropic's ceiling of 1.0. Narrowing is the adapter's obligation.
    assert _request(temperature=1.5).temperature == 1.5


def test_a_model_name_past_the_limit_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(
            call_id=new_call_id(),
            model="m" * (MAX_MODEL_NAME_LENGTH + 1),
            messages=(_user(),),
            max_tokens=16,
        )


def test_a_result_carries_the_model_that_actually_served_it() -> None:
    call_id = new_call_id()
    result = CompletionResult(
        call_id=call_id,
        model="claude-opus-5-20260101",
        text="A comparison.",
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=1200, output_tokens=300),
        latency=timedelta(milliseconds=900),
    )
    assert result.call_id == call_id
    assert result.model == "claude-opus-5-20260101"
    assert result.usage.total == 1500


def test_an_empty_completion_is_a_real_outcome() -> None:
    # Requiring content would only push an adapter into inventing some.
    result = CompletionResult(
        call_id=new_call_id(),
        model="claude-opus-5",
        text="",
        stop_reason=StopReason.STOP_SEQUENCE,
        usage=TokenUsage(input_tokens=10, output_tokens=0),
        latency=timedelta(milliseconds=40),
    )
    assert result.text == ""


def test_truncation_is_visible_in_the_type() -> None:
    result = CompletionResult(
        call_id=new_call_id(),
        model="claude-opus-5",
        text="A comparison that was cut off",
        stop_reason=StopReason.MAX_TOKENS,
        usage=TokenUsage(input_tokens=1200, output_tokens=1024),
        latency=timedelta(seconds=3),
    )
    assert result.stop_reason is StopReason.MAX_TOKENS


def test_a_result_carries_no_cost() -> None:
    # Cost is computed from usage against a versioned price list, which is the next
    # increment. This test exists so that putting a price here is a deliberate act.
    assert "amount" not in CompletionResult.model_fields
    assert "cost" not in CompletionResult.model_fields


def test_a_negative_latency_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CompletionResult(
            call_id=new_call_id(),
            model="claude-opus-5",
            text="",
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            latency=timedelta(milliseconds=-1),
        )


def test_roles_and_stop_reasons_render_as_their_labels() -> None:
    assert str(Role.ASSISTANT) == "assistant"
    assert str(StopReason.MAX_TOKENS) == "max_tokens"
