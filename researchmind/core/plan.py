"""The plan: the typed decomposition a run executes, and the artefact a user approves.

The plan is where the system's intent becomes legible before any money is spent (design
principle 1). It is also the spine of the trace: a sub-question identifier appears in
every event, every cost row and every extracted fact (ADR-0002), which is why these types
settle before the rest of the domain does.

Three consequences of ADR-0002 are load-bearing here.

A plan is a **revision**, never a mutable document. An edit produces revision ``N + 1``
and leaves its predecessor intact, so a sub-question identifier already recorded in the
trace can never come to mean something else. This module therefore supplies no state
transitions: approving, rejecting and revising are planner behaviour (phase 7), not
properties of the shape.

Dependencies form a **directed acyclic graph**, and acyclicity is checked rather than
guaranteed by construction. Restricting a step to depend only on earlier steps would make
cycles impossible, but it would also require the planner to emit a topological order it
has no reason to know. The order of ``steps`` is presentation order — the ``ordinal``
column of ADR-0002, assigned by ``storage`` from position — and carries no dependency
meaning at all.

Expected tools are **strings, not an enumeration**. The tool registry arrives in phase 5,
and ``core`` imports nothing of ours. A name is checked against the registry during plan
validation, where the registry exists; encoding it here would reverse the dependency for
the sake of convenience.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Final, Self

from pydantic import Field, model_validator

from researchmind.core.base import DomainModel
from researchmind.core.clock import UtcDatetime
from researchmind.core.ids import PlanId, RunId, SubQuestionId

MAX_SUB_QUESTION_LENGTH: Final = 500
"""Longest sub-question text, in characters."""

MAX_RATIONALE_LENGTH: Final = 1000
"""Longest rationale, in characters."""

MAX_TOOL_NAME_LENGTH: Final = 64
"""Longest tool name, in characters. A name is an identifier, not a description."""

MAX_EXPECTED_TOOLS: Final = 8
"""Most tools a single step may nominate."""

MAX_STEPS: Final = 64
"""Most steps a plan may contain.

A bound exists because every step costs tokens and wall-clock time, and a plan of three
hundred steps is a runaway decomposition rather than a thorough one. It is a guard rail,
not a target.
"""

ToolName = Annotated[str, Field(min_length=1, max_length=MAX_TOOL_NAME_LENGTH)]
"""The name of a tool, validated against the registry in phase 5, not here."""


class PlanStatus(Enum):
    """Where a plan revision stands in review.

    There is no ``superseded``: a revision is superseded exactly when the same run has a
    higher revision, which is a query rather than a second source of truth that could
    disagree with the first.
    """

    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``approved``."""
        return self.value


class PlanAuthor(Enum):
    """Who produced a plan revision — the ``created_by`` column of ADR-0002.

    A revision written by a user edit must stay distinguishable from one the planner
    proposed; without it the revision history reads as if the machine changed its mind.
    """

    PLANNER = "planner"
    USER = "user"

    def __str__(self) -> str:
        """Render as the stored label, so logs read ``planner``."""
        return self.value


class SubQuestion(DomainModel):
    """One step of a plan: a question to answer, and why it is being asked.

    The rationale is mandatory and non-empty on purpose. The plan exists to be read and
    approved before the first tool call is paid for, and a step that does not say why it
    exists cannot be reviewed — only waved through.
    """

    id: SubQuestionId
    text: str = Field(min_length=1, max_length=MAX_SUB_QUESTION_LENGTH)
    rationale: str = Field(min_length=1, max_length=MAX_RATIONALE_LENGTH)
    expected_tools: tuple[ToolName, ...] = Field(default=(), max_length=MAX_EXPECTED_TOOLS)
    depends_on: tuple[SubQuestionId, ...] = ()

    @model_validator(mode="after")
    def _reject_degenerate_edges(self) -> Self:
        """Reject a step that repeats itself or waits for itself.

        Both are cheap to state here, where only one step is in scope, and both would
        otherwise have to be untangled from a cycle report covering the whole graph.
        """
        if len(set(self.depends_on)) != len(self.depends_on):
            msg = "depends_on lists the same sub-question more than once"
            raise ValueError(msg)
        if self.id in self.depends_on:
            msg = "a sub-question cannot depend on itself"
            raise ValueError(msg)
        if len(set(self.expected_tools)) != len(self.expected_tools):
            msg = "expected_tools names the same tool more than once"
            raise ValueError(msg)
        return self


class Plan(DomainModel):
    """An immutable plan revision: the ordered steps a run intends to execute.

    Validation covers the whole graph, so a ``Plan`` that exists is one whose steps are
    uniquely identified, whose dependencies all resolve, and whose dependency graph can
    actually be executed.
    """

    id: PlanId
    run_id: RunId
    revision: int = Field(ge=1)
    status: PlanStatus
    created_by: PlanAuthor
    created_at: UtcDatetime
    steps: tuple[SubQuestion, ...] = Field(min_length=1, max_length=MAX_STEPS)

    @model_validator(mode="after")
    def _validate_step_graph(self) -> Self:
        """Check the three invariants that span steps, in the order they depend on.

        Duplicate identifiers are found first because everything after them would be
        reasoning about an ambiguous graph, and dangling references before cycles because
        cycle detection assumes every edge points at a step that exists.
        """
        known = _reject_duplicate_ids(self.steps)
        _reject_dangling_dependencies(self.steps, known)
        _reject_cycles(self.steps)
        return self


def _reject_duplicate_ids(steps: tuple[SubQuestion, ...]) -> frozenset[SubQuestionId]:
    """Return the set of step identifiers, refusing a plan that reuses one.

    A repeated identifier would let a fact, a cost row or an event point at two different
    steps, which makes the trace ambiguous exactly where it is supposed to be decisive.
    """
    known = frozenset(step.id for step in steps)
    if len(known) != len(steps):
        msg = "two steps share a sub-question identifier"
        raise ValueError(msg)
    return known


def _reject_dangling_dependencies(
    steps: tuple[SubQuestion, ...], known: frozenset[SubQuestionId]
) -> None:
    """Refuse a plan whose steps wait on sub-questions that are not in it.

    A dependency on an absent step is not a slow step, it is a step that never becomes
    ready. Caught here, it is a validation error on a plan nobody approved yet.
    """
    for step in steps:
        missing = sorted(str(other) for other in step.depends_on if other not in known)
        if missing:
            msg = (
                f"step {step.id} depends on sub-questions absent from the plan: "
                f"{', '.join(missing)}"
            )
            raise ValueError(msg)


def _reject_cycles(steps: tuple[SubQuestion, ...]) -> None:
    """Refuse a plan whose dependencies form a cycle, by Kahn's algorithm.

    Steps with nothing left to wait for are settled repeatedly; whatever never settles is
    part of a cycle or waits on one. The scan over unsettled steps makes this quadratic
    in the worst case, which at the sixty-four-step ceiling is not worth an index to
    avoid.
    """
    outstanding = {step.id: set(step.depends_on) for step in steps}
    ready = [step_id for step_id, pending in outstanding.items() if not pending]
    settled = 0

    while ready:
        step_id = ready.pop()
        del outstanding[step_id]
        settled += 1
        for other, pending in outstanding.items():
            if step_id in pending:
                pending.discard(step_id)
                if not pending:
                    ready.append(other)

    if settled != len(steps):
        stuck = sorted(str(step_id) for step_id in outstanding)
        msg = f"steps depend on each other in a cycle: {', '.join(stuck)}"
        raise ValueError(msg)
