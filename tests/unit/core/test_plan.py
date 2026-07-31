"""Worked examples of plans that hold together, and of the ones that must not.

Every invariant stated in ``core.plan`` gets a case that violates it, because an
invariant nothing tests is a comment.
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from researchmind.core.clock import utc_now
from researchmind.core.ids import (
    SubQuestionId,
    new_plan_id,
    new_run_id,
    new_sub_question_id,
)
from researchmind.core.plan import (
    MAX_STEPS,
    Plan,
    PlanAuthor,
    PlanStatus,
    SubQuestion,
)


def _step(
    step_id: SubQuestionId | None = None,
    *,
    text: str = "Which jurisdictions issued stablecoin rules in 2025?",
    rationale: str = "Establishes the baseline the later comparisons are drawn against.",
    expected_tools: tuple[str, ...] = (),
    depends_on: tuple[SubQuestionId, ...] = (),
) -> SubQuestion:
    return SubQuestion(
        id=step_id if step_id is not None else new_sub_question_id(),
        text=text,
        rationale=rationale,
        expected_tools=expected_tools,
        depends_on=depends_on,
    )


def _plan(
    steps: tuple[SubQuestion, ...],
    *,
    revision: int = 1,
    status: PlanStatus = PlanStatus.DRAFT,
    created_by: PlanAuthor = PlanAuthor.PLANNER,
    created_at: datetime | None = None,
) -> Plan:
    return Plan(
        id=new_plan_id(),
        run_id=new_run_id(),
        revision=revision,
        status=status,
        created_by=created_by,
        created_at=created_at if created_at is not None else utc_now(),
        steps=steps,
    )


def test_a_plan_with_dependencies_is_accepted() -> None:
    first = _step(expected_tools=("web_search", "web_fetch"))
    second = _step(depends_on=(first.id,))
    third = _step(depends_on=(first.id, second.id))

    plan = _plan((first, second, third))

    assert plan.steps == (first, second, third)
    assert plan.revision == 1
    assert plan.status is PlanStatus.DRAFT


def test_dependencies_need_not_point_backwards() -> None:
    # The order of steps is presentation order, not execution order (ADR-0002). A plan
    # listing a step before the one it waits for is unusual, not invalid.
    later = new_sub_question_id()
    first = _step(depends_on=(later,))
    second = _step(later)

    assert _plan((first, second)).steps == (first, second)


def test_a_plan_needs_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        _plan(())


def test_a_plan_past_the_step_ceiling_is_rejected() -> None:
    assert len(_plan(tuple(_step() for _ in range(MAX_STEPS))).steps) == MAX_STEPS
    with pytest.raises(ValidationError):
        _plan(tuple(_step() for _ in range(MAX_STEPS + 1)))


def test_two_steps_cannot_share_an_identifier() -> None:
    shared = new_sub_question_id()
    with pytest.raises(ValidationError, match="share a sub-question identifier"):
        _plan((_step(shared), _step(shared)))


def test_a_dependency_on_an_absent_step_is_rejected() -> None:
    with pytest.raises(ValidationError, match="absent from the plan"):
        _plan((_step(depends_on=(new_sub_question_id(),)),))


def test_a_step_cannot_depend_on_itself() -> None:
    step_id = new_sub_question_id()
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        _step(step_id, depends_on=(step_id,))


def test_a_two_step_cycle_is_rejected() -> None:
    first_id = new_sub_question_id()
    second_id = new_sub_question_id()
    first = _step(first_id, depends_on=(second_id,))
    second = _step(second_id, depends_on=(first_id,))

    with pytest.raises(ValidationError, match="cycle"):
        _plan((first, second))


def test_a_cycle_reachable_from_a_valid_step_is_rejected() -> None:
    # The cycle sits behind a step that is perfectly settleable, so a check that stopped
    # at the first ready step would miss it.
    root = _step()
    left_id = new_sub_question_id()
    right_id = new_sub_question_id()
    left = _step(left_id, depends_on=(root.id, right_id))
    right = _step(right_id, depends_on=(left_id,))

    with pytest.raises(ValidationError, match="cycle"):
        _plan((root, left, right))


def test_a_step_cannot_list_the_same_dependency_twice() -> None:
    other = new_sub_question_id()
    with pytest.raises(ValidationError, match="more than once"):
        _step(depends_on=(other, other))


def test_a_step_cannot_name_the_same_tool_twice() -> None:
    with pytest.raises(ValidationError, match="same tool more than once"):
        _step(expected_tools=("web_search", "web_search"))


def test_a_tool_name_cannot_be_blank() -> None:
    with pytest.raises(ValidationError):
        _step(expected_tools=("",))


def test_a_step_may_nominate_no_tools() -> None:
    # A step that only reasons over what earlier steps found makes no tool call.
    assert _step().expected_tools == ()


@pytest.mark.parametrize("rationale", ["", "   "])
def test_a_step_without_a_rationale_is_rejected(rationale: str) -> None:
    # The rationale is what a reviewer reads when deciding whether to approve the plan.
    with pytest.raises(ValidationError):
        _step(rationale=rationale)


def test_a_step_without_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _step(text="  ")


def test_revisions_start_at_one() -> None:
    with pytest.raises(ValidationError):
        _plan((_step(),), revision=0)


def test_the_creation_instant_must_be_aware() -> None:
    with pytest.raises(ValidationError):
        _plan((_step(),), created_at=datetime(2026, 7, 28, 12, 0))  # noqa: DTZ001


def test_the_creation_instant_is_normalised_to_utc() -> None:
    moscow = timezone(timedelta(hours=3))
    plan = _plan((_step(),), created_at=datetime(2026, 7, 28, 15, 0, tzinfo=moscow))
    assert plan.created_at.utcoffset() == timedelta(0)


def test_collections_arrive_as_tuples() -> None:
    # Freezing protects attributes, not containers; the domain uses tuples so that a
    # frozen model is immutable all the way down.
    plan = _plan((_step(expected_tools=("web_search",)),))
    assert isinstance(plan.steps, tuple)
    assert isinstance(plan.steps[0].expected_tools, tuple)
    assert isinstance(plan.steps[0].depends_on, tuple)


def test_a_plan_carries_no_state_transitions() -> None:
    # Approving or revising is planner behaviour (phase 7). A plan revision is a record,
    # and this test exists so that adding a mutator here is a deliberate act.
    plan = _plan((_step(),))
    with pytest.raises(ValidationError):
        plan.status = PlanStatus.APPROVED  # type: ignore[misc]


def test_statuses_and_authors_render_as_their_labels() -> None:
    assert str(PlanStatus.APPROVED) == "approved"
    assert str(PlanAuthor.USER) == "user"
