"""Properties of the call contract, over generated exchanges and the whole taxonomy.

The alternation rule is the kind of invariant worth generating: it is easy to state, easy
to implement subtly wrong at the boundaries, and every counterexample is a request that
would have failed at a vendor instead of at construction.
"""

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from researchmind.core.errors import ResearchmindError
from researchmind.core.ids import new_call_id
from researchmind.providers.completion import CompletionRequest, Message, Role
from researchmind.providers.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

ROLE_SEQUENCES = st.lists(st.sampled_from(list(Role)), min_size=1, max_size=8)
CONTENTS = st.text(min_size=1, max_size=60).map(str.strip).filter(bool)
TEMPERATURES = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
PROVIDER_ERRORS = st.sampled_from(
    [
        ProviderAuthError,
        ProviderRequestError,
        ProviderResponseError,
        ProviderRateLimitError,
        ProviderUnavailableError,
        ProviderTimeoutError,
    ]
)


def _request(messages: tuple[Message, ...], temperature: float = 0.0) -> CompletionRequest:
    return CompletionRequest(
        call_id=new_call_id(),
        model="claude-opus-5",
        messages=messages,
        max_tokens=256,
        temperature=temperature,
    )


@given(ROLE_SEQUENCES)
def test_an_exchange_is_accepted_exactly_when_it_alternates_from_the_user(
    roles: list[Role],
) -> None:
    alternates = roles[0] is Role.USER and all(
        earlier is not later for earlier, later in pairwise(roles)
    )
    messages = tuple(Message(role=role, content="a turn") for role in roles)

    if alternates:
        assert [message.role for message in _request(messages).messages] == roles
    else:
        with pytest.raises(ValidationError):
            _request(messages)


@given(st.lists(CONTENTS, min_size=1, max_size=6), TEMPERATURES)
def test_a_request_keeps_what_it_was_given(contents: list[str], temperature: float) -> None:
    # Turns are not reordered and the system prompt is not folded into the exchange. What
    # the adapter receives is what the caller wrote.
    roles = [Role.USER if index % 2 == 0 else Role.ASSISTANT for index in range(len(contents))]
    messages = tuple(
        Message(role=role, content=content) for role, content in zip(roles, contents, strict=True)
    )

    request = _request(messages, temperature)

    assert [message.content for message in request.messages] == contents
    assert [message.role for message in request.messages] == roles
    assert request.temperature == temperature
    assert request.system is None


@given(PROVIDER_ERRORS)
def test_every_failure_in_the_taxonomy_is_catchable_as_ours(
    error_type: type[ProviderError],
) -> None:
    with pytest.raises(ProviderError):
        raise error_type("failed", provider="anthropic")

    with pytest.raises(ResearchmindError):
        raise error_type("failed", provider="anthropic")


@given(PROVIDER_ERRORS)
def test_every_failure_carries_the_provider_that_produced_it(
    error_type: type[ProviderError],
) -> None:
    error = error_type("failed", provider="vllm", status=500)
    assert error.provider == "vllm"
    assert error.status == 500
