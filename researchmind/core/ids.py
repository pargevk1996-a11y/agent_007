"""Identifiers for every entity the system names.

Each identifier is a distinct type to mypy and a plain :class:`uuid.UUID` at runtime, so
handing a ``FactId`` to something expecting a ``SourceId`` is a type error rather than a
debugging session. There is no runtime cost: ``NewType`` erases at import time.

Values are UUIDv7 (ADR-0009). The leading 48 bits are a millisecond timestamp, so
identifiers increase over time — they cluster in a b-tree index instead of scattering,
and ``ORDER BY id`` is chronological to the millisecond.

That last qualifier is load-bearing. Two identifiers minted inside the same millisecond
have no defined order between them, because the remaining bits are random. Anything that
needs a total order over events uses the per-run sequence number instead (ADR-0008).

Identifiers are minted by the application, never by the database, because an object is
logged before it is written and a failed write is precisely the case worth tracing.
"""

from typing import NewType
from uuid import UUID

# The top-level uuid_utils returns its own Rust-backed UUID class, which is not a
# uuid.UUID and which pydantic rejects. The compat module returns the standard one.
from uuid_utils.compat import uuid7

# Identities we receive rather than create. They arrive already verified, from whatever
# mechanism the control plane uses, so this module deliberately offers no factory for
# them: minting a tenant here would mean the domain had invented an identity.
TenantId = NewType("TenantId", UUID)
UserId = NewType("UserId", UUID)

# Identities we mint.
RunId = NewType("RunId", UUID)
PlanId = NewType("PlanId", UUID)
SubQuestionId = NewType("SubQuestionId", UUID)
SourceId = NewType("SourceId", UUID)
FactId = NewType("FactId", UUID)
ClaimId = NewType("ClaimId", UUID)
CallId = NewType("CallId", UUID)


def new_run_id() -> RunId:
    """Mint an identifier for a research run."""
    return RunId(uuid7())


def new_plan_id() -> PlanId:
    """Mint an identifier for a plan revision."""
    return PlanId(uuid7())


def new_sub_question_id() -> SubQuestionId:
    """Mint an identifier for a sub-question."""
    return SubQuestionId(uuid7())


def new_source_id() -> SourceId:
    """Mint an identifier for a retrieved source."""
    return SourceId(uuid7())


def new_fact_id() -> FactId:
    """Mint an identifier for an extracted fact."""
    return FactId(uuid7())


def new_claim_id() -> ClaimId:
    """Mint an identifier for a claim in a report."""
    return ClaimId(uuid7())


def new_call_id() -> CallId:
    """Mint an identifier for a single LLM or tool call."""
    return CallId(uuid7())
