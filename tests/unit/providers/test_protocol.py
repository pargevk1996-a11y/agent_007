"""Conformance to the provider protocol, checked by types and exercised once.

The assertion that matters here is made by ``mypy --strict``, not at runtime: passing
``StubProvider`` to a parameter annotated ``LLMProvider`` type-checks only if the class
really has the shape. The runtime test then proves the shape is usable rather than merely
declared.
"""

from datetime import timedelta

import pytest

from researchmind.core.ids import new_call_id
from researchmind.core.tokens import TokenUsage
from researchmind.providers.base import LLMProvider, StructuredOutputMode
from researchmind.providers.completion import (
    CompletionRequest,
    CompletionResult,
    Message,
    Role,
    StopReason,
)


class StubProvider:
    """A provider that answers without a network, conforming by shape alone."""

    @property
    def name(self) -> str:
        """Return the name recorded against every cost row."""
        return "stub"

    @property
    def structured_output_mode(self) -> StructuredOutputMode:
        """Return the strongest guarantee this stub pretends to offer."""
        return StructuredOutputMode.NATIVE_TOOLS

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Echo the request back as a completion."""
        return CompletionResult(
            call_id=request.call_id,
            model=request.model,
            text=f"answered: {request.messages[-1].content}",
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency=timedelta(milliseconds=1),
        )


def _use(provider: LLMProvider) -> LLMProvider:
    """Accept anything satisfying the protocol; this call is what mypy verifies."""
    return provider


def test_a_class_of_the_right_shape_satisfies_the_protocol() -> None:
    provider = _use(StubProvider())
    assert provider.name == "stub"
    assert provider.structured_output_mode is StructuredOutputMode.NATIVE_TOOLS


async def test_the_protocol_describes_a_call_that_actually_runs() -> None:
    provider = _use(StubProvider())
    request = CompletionRequest(
        call_id=new_call_id(),
        model="claude-opus-5",
        messages=(Message(role=Role.USER, content="Which regulators published in 2025?"),),
        max_tokens=256,
    )

    result = await provider.complete(request)

    assert result.call_id == request.call_id
    assert result.text.endswith("Which regulators published in 2025?")


def test_conformance_is_checked_by_types_and_not_at_runtime() -> None:
    # The protocol is deliberately not runtime_checkable: an isinstance check over a
    # protocol tests for the presence of attributes and would call a provider with a
    # wrongly typed `complete` conforming. This documents the limitation as a fact.
    with pytest.raises(TypeError):
        isinstance(StubProvider(), LLMProvider)  # type: ignore[misc]


def test_the_declared_modes_are_the_ones_the_decision_names() -> None:
    assert {str(mode) for mode in StructuredOutputMode} == {
        "native_tools",
        "json_schema",
        "guided_decoding",
        "prompted",
    }
