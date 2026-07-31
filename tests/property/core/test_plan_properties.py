"""Plan validation checked over generated graphs rather than over three drawn examples.

ADR-0002 asks for acyclicity to be covered by a property-based test, and the reason is
that cycles are exactly the failure a handful of examples misses: the shapes that break a
naive check are the ones nobody thinks to write down.

Identifiers here are drawn by hypothesis rather than minted with ``new_sub_question_id``.
Minted values would be fresh on every run, so a failing example could not be replayed and
shrinking would have nothing stable to work with. The version of the UUID is irrelevant
to the invariants under test — only its identity is.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.ids import SubQuestionId, new_plan_id, new_run_id
from researchmind.core.plan import Plan, PlanAuthor, PlanStatus, SubQuestion

MAX_GENERATED_STEPS = 8
"""Deliberately far below MAX_STEPS: these properties are about shape, not scale."""

IDENTIFIERS = st.uuids().map(SubQuestionId)


def _step(step_id: SubQuestionId, depends_on: tuple[SubQuestionId, ...] = ()) -> SubQuestion:
    return SubQuestion(
        id=step_id,
        text="A generated sub-question.",
        rationale="Generated: the shape of the graph is what is under test.",
        depends_on=depends_on,
    )


def _plan(steps: tuple[SubQuestion, ...]) -> Plan:
    return Plan(
        id=new_plan_id(),
        run_id=new_run_id(),
        revision=1,
        status=PlanStatus.DRAFT,
        created_by=PlanAuthor.PLANNER,
        created_at=utc_now(),
        steps=steps,
    )


@st.composite
def acyclic_steps(draw: st.DrawFn) -> tuple[SubQuestion, ...]:
    """Build a random DAG by letting each step depend only on the ones drawn before it."""
    identifiers = draw(st.lists(IDENTIFIERS, min_size=1, max_size=MAX_GENERATED_STEPS, unique=True))
    steps = []
    for index, step_id in enumerate(identifiers):
        earlier = identifiers[:index]
        depends_on = (
            draw(st.lists(st.sampled_from(earlier), unique=True, max_size=len(earlier)))
            if earlier
            else []
        )
        steps.append(_step(step_id, tuple(depends_on)))
    return tuple(steps)


@st.composite
def shuffled_acyclic_steps(draw: st.DrawFn) -> tuple[SubQuestion, ...]:
    """Draw a DAG and present it in an arbitrary order."""
    steps = draw(acyclic_steps())
    return tuple(draw(st.permutations(list(steps))))


@st.composite
def cyclic_steps(draw: st.DrawFn) -> tuple[SubQuestion, ...]:
    """Build a closed loop: every step waits on its predecessor, the first on the last."""
    identifiers = draw(st.lists(IDENTIFIERS, min_size=2, max_size=MAX_GENERATED_STEPS, unique=True))
    steps = [_step(step_id, (identifiers[index - 1],)) for index, step_id in enumerate(identifiers)]
    return tuple(draw(st.permutations(steps)))


@st.composite
def steps_with_a_dangling_dependency(draw: st.DrawFn) -> tuple[SubQuestion, ...]:
    """Take a valid DAG and point one of its steps at a sub-question that is not in it."""
    steps = list(draw(acyclic_steps()))
    present = {step.id for step in steps}
    absent = draw(IDENTIFIERS.filter(lambda candidate: candidate not in present))
    index = draw(st.integers(min_value=0, max_value=len(steps) - 1))
    victim = steps[index]
    steps[index] = _step(victim.id, (*victim.depends_on, absent))
    return tuple(steps)


@given(acyclic_steps())
def test_any_acyclic_plan_is_accepted(steps: tuple[SubQuestion, ...]) -> None:
    assert _plan(steps).steps == steps


@given(shuffled_acyclic_steps())
def test_presentation_order_does_not_decide_validity(steps: tuple[SubQuestion, ...]) -> None:
    # The order of steps is the ordinal of ADR-0002 — how the plan is shown, not how it
    # is executed. A DAG stays valid however it is arranged, which is what lets the
    # planner emit steps in whatever order it finds natural.
    assert _plan(steps).steps == steps


@given(cyclic_steps())
def test_a_cycle_is_always_rejected(steps: tuple[SubQuestion, ...]) -> None:
    with pytest.raises(ValidationError, match="cycle"):
        _plan(steps)


@given(steps_with_a_dangling_dependency())
def test_a_dependency_outside_the_plan_is_always_rejected(
    steps: tuple[SubQuestion, ...],
) -> None:
    with pytest.raises(ValidationError, match="absent from the plan"):
        _plan(steps)
